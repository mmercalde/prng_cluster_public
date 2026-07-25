# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md — REV3 (rebased onto `c207e3a`)

**REV3 changelog — rebase only, no scope change.** All citations re-verified
against HEAD `c207e3a` (post-D3). Corrections: `is_var` is at `:279` (was cited
`:278`); the **third** adapter call site is ZMQ at `:525` (previously named but
uncited); **PWC has its own pruned early-return at `:1621-1629`** carrying no
map keys at all, which needs the v2 shape exactly as ZMQ's `:1095-1098` does
(previously only ZMQ's was cited). One substantive addition, §5a: **D3 now
provides an executable definition of a valid canonical 24-field record**, so
D3.25's normalizer output must pass it — a free conformance gate that did not
exist when REV2 was written. `_mode_records` confirmed still at
`miner/range_miner_npz_writer.py:432`.

**S172 RANGE-MINER — Phase 5, Deliverable D3.25: mode-preserving backend
result contract + canonical candidate-ingress normalization**

**REV2 changelog** — absorbs the Team Beta pre-implementation review
(APPROVED WITH REQUIRED CORRECTIONS), six items: **[C1]** corrected ingress
topology (the adapter does NOT serve the original legacy path); **[C2]**
`utils/canonical_records.py` must not depend on `WindowConfig`; **[C3]**
`CANONICAL_RECORD_FIELDS` moves with the helper; **[C4]** v2 contract validated
at BOTH producer egress and adapter ingress; **[C5]** D1's existing `sessions`
semantics preserved through the extraction; **[C6]** expanded mutation set.
Plus gate amendments G1/G4/G6/G9/G11 and explicit scope boundaries.

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push,
or run WATCHER. When gates + non-regression are green, STOP and report.

**Base: HEAD `c207e3a`** (D3 committed). Every citation in this document was re-verified against that commit; D3.0 and D3 have both landed, so no further rebase is pending.

---

## 0. The defect, verified at source

### 0.1 Ingress topology — three append sites, not one **[C1]**

`survivor_accumulator['bidirectional']` is appended at exactly three places in
`window_optimizer_integration_final.py`:

| site | path | behavior |
|---|---|---|
| `:281` | `_build_test_result_from_pw` — **PWC (`:472`), ZMQ (`:525`), and the currently miswired miner path (`:426`)** | set **union** of both modes → ONE record |
| `:686` | original legacy **constant** block | appends constant candidates directly |
| `:788` | original legacy **variable** block | appends variable candidates directly |

So `_build_test_result_from_pw` is the shared ingress for PWC, ZMQ and the
current miner placeholder. **The original legacy sieve path has an independent
direct candidate-construction path** and remains a Boundary-1 comparison
participant — valuable precisely because it prevents every Boundary-1 path from
exercising the same record-construction implementation.

**The legacy path is the correct reference implementation.** It appends the two
modes independently at two separate sites, with per-mode maps and per-mode
aggregates — which is why D1.1's formulas were frozen from those exact blocks
(`:652-694` constant, `:756-796` variable). The adapter's own docstring
(`:258-259`) claims it *"Mirrors the accumulator logic in the original
run_bidirectional_test path."* **It does not.** Legacy appends twice; the
adapter unions into one. D3.25 makes the docstring true.

### 0.2 The four defects in the adapter

1. **Cross-mode collapse.** `for seed in bidi_constant | bidi_variable`
   (`:276`) with `is_var = seed in bidi_variable` (`:279`): a seed in BOTH
   populations yields ONE record labelled variable — the constant candidate is
   destroyed before the L2 competition boundary. D1/D2 established cross-mode
   duplication is legitimate.
2. **Mode-agnostic rates.** The adapter receives ONE `forward_map` / ONE
   `reverse_map` (`:265-266`); `fmr`/`rmr` (`:277-278`) are constant-mode rates
   regardless of the record's mode. Splitting the union alone would emit a
   variable record carrying constant rates.
3. **Constant-biased aggregates.** `intersection_ratio`,
   `survivor_overlap_ratio`, `intersection_weight` (`:300-302`) use
   `len(bidi_constant)` and the generic maps for every record. This corrupts
   the ML feature surface.
4. **Format divergence.** `skip_range` is the **string**
   `f"{config.skip_min}-{config.skip_max}"` (`:304`) where D1.1 emits `int`;
   `sessions` uses `getattr(config, 'sessions', 'all')` (`:303`), which can
   yield the scalar `"all"` where D1.1 emits a list.

### 0.3 Root cause — the producers discard the information

- **PWC** computes `fwd_h_map` (`:1690`), `rev_h_map` (`:1717`), derives
  `bidirectional_variable = set(fwd_h_map) & set(rev_h_map)` (`:1720`) — then
  returns only `forward_map`/`reverse_map`, the **constant** pair
  (`:1752`). Both variable maps are structurally discarded. PWC also has a
  **pruned early-return at `:1621-1629`** (`reason: "forward_zero"`) carrying
  no map keys whatsoever — the adapter's `.get("forward_map", {})` silently
  supplies `{}`.
- **ZMQ** identical (`:1136`, `:1151`, `:1155`; return `:1186`), plus an
  early/pruned path (`:1095-1098`) returning only
  `"forward_map": {}, "reverse_map": {}` — no variable keys at all.

**The record lists are NOT a substitute (binding).** Both backends return
`forward_records_hybrid`/`reverse_records_hybrid`, but their provenance
differs: PWC builds from the raw survivor sequence (`for s in fwd_h_survivors`,
`:1723`), ZMQ from map keys (`for s in fwd_h_map`, `:1156`). A repeated raw
seed makes list and map inequivalent. **The four explicit maps are
authoritative; record lists are telemetry/compatibility data only; no canonical
adapter may reconstruct a map from a record list.**

## 1. Non-negotiable working rules

1. **Read live source before every claim** — re-verify every cite after
   rebasing onto post-D3.0 HEAD.
2. **Each gate must FAIL on wrong behavior.** The pre-fix adapter MUST fail
   G2, G3, G5, G6; demonstrate this before editing.
3. **Semantics-preserving extraction.** D1/D2 gates staying green is the proof
   (§4).
4. STOP at the gate. No commit/push/WATCHER.

## 2. D3.25-A — versioned producer contract, validated at BOTH boundaries **[C4]**

Required return shape from PWC and ZMQ:

```python
{
    "schema_version": "step1_trial_populations_v2",
    "forward_map_constant":   dict[int, float],
    "reverse_map_constant":   dict[int, float],
    "forward_map_variable":   dict[int, float],
    "reverse_map_variable":   dict[int, float],
    "bidirectional_constant": set[int],
    "bidirectional_variable": set[int],
    "pruned": bool,
    "reason": str | None,
}
```

- **Shape never varies.** Constant-only trials, both-mode trials,
  forward-zero pruned returns, and any other supported pruned return all carry
  the complete shape with all four maps present (empty where the mode did not
  run). **Missing fields are NOT interpreted as empty fields** — including
  ZMQ's early return at `:1095-1098` **and** PWC's at `:1621-1629`.
- Legacy `forward_map`/`reverse_map` aliases may remain temporarily; the v2
  adapter must NOT read them (G7).

**Validation happens twice — this is what makes G4 meaningful [C4]:**

*Producer egress* — PWC and ZMQ assert immediately before returning:

```python
bidirectional_constant == set(forward_map_constant) & set(reverse_map_constant)
bidirectional_variable == set(forward_map_variable) & set(reverse_map_variable)
```

*Adapter ingress* — `_build_test_result_from_pw` independently validates
exact `schema_version`; presence and types of all four maps; presence and types
of both bidirectional sets; both set/map intersection equalities; the
pruned-shape rules. **Only after ingress validation succeeds may the
accumulator be modified** — a malformed or test-mutated result must fail before
even one candidate is appended.

Neither boundary may silently repair a disagreement by preferring the returned
set or a recomputed intersection: a mismatch is producer-state corruption and
fails closed.

## 3. D3.25-B — the one shared canonical normalizer

**Placement (approved): `utils/canonical_records.py`.** Dependency direction is
generic utilities ← {miner, PWC, ZMQ, adapter}. PWC/ZMQ must NOT import record
construction from `miner/`.

**Extraction target:** `_mode_records` at `miner/range_miner_npz_writer.py:432`
— already the exact required shape. Move it **unchanged in semantics**.

### 3.1 Public surface — no `WindowConfig` dependency **[C2]**

The utility must not depend on the `WindowConfig` class and must not use
generic attribute lookup internally. D1 receives validated manifest metadata;
PWC/ZMQ receive a `WindowConfig`; the shared utility sits **below** both.

```python
def build_mode_records(
    forward_map: Mapping[int, float],
    reverse_map: Mapping[int, float],
    context: Mapping[str, object],
    skip_mode: str,
    prng_type: str | None,
) -> tuple[set[int], list[dict]]:
    """Semantics-preserving extraction of _mode_records."""

def normalize_trial_populations(
    forward_map_constant, reverse_map_constant,
    forward_map_variable, reverse_map_variable,
    *, window_size, offset, skip_min, skip_max, sessions,
    trial_number, prng_base,
) -> tuple[list[dict], list[dict]]:
    """(constant_records, variable_records). Explicit values only."""
```

The **adapter** is responsible for reading mandatory `WindowConfig` attributes
directly and passing validated values in — never a config object.

### 3.2 `CANONICAL_RECORD_FIELDS` moves with the helper **[C3]**

Move the production constant into `utils/canonical_records.py` beside the
builder; `miner/range_miner_npz_writer.py` imports both
`CANONICAL_RECORD_FIELDS` and `build_mode_records` from it. **D1.1's G9 harness
must continue comparing the production constant against its independently
hand-transcribed oracle** — relocating the constant does NOT authorize
importing it into the test oracle.

### 3.3 Per-mode derivation

For mode `M`, computed ONLY from that mode's maps, formulas and `max(..., 1)`
guards identical to D1.1:

```python
fwd, rev = forward_map_M, reverse_map_M
both  = set(fwd) & set(rev)
union = set(fwd) | set(rev)
forward_count = len(fwd); reverse_count = len(rev)
bidirectional_count = intersection_count = len(both)
intersection_ratio        = len(both) / max(len(union), 1)
forward_only_count        = len(set(fwd) - set(rev))
reverse_only_count        = len(set(rev) - set(fwd))
survivor_overlap_ratio    = len(both) / max(len(fwd), 1)
bidirectional_selectivity = len(fwd) / max(len(rev), 1)
intersection_weight       = len(both) / max(len(fwd) + len(rev), 1)
score = (fwd[seed] + rev[seed]) / 2.0
```

**No combined constant+variable count belongs in a record.** Combined totals
remain `TestResult`/dashboard telemetry only.

### 3.4 Canonical field forms, and the `sessions` preservation rule **[C5]**

- `skip_range = int(skip_max) - int(skip_min)`. The string form is
  **prohibited** for new candidates.
- `sessions`: list/tuple → defensive list copy; `None` → `[]`; attribute
  absent → **fail closed**; scalar string such as `"all"` → **fail closed**
  (do NOT convert to `["all"]`).
- **Extraction constraint:** `_mode_records` today places the context's
  `sessions` object directly into each record (shared reference — a D1.1
  behavior Team Beta accepted as non-blocking). **Do not silently deep-copy or
  normalize D1's already-validated context during the extraction.** The
  PWC/ZMQ *wrapper* may defensively normalize its own incoming `sessions`
  before constructing the context it passes down. Changing D1's behavior
  requires a separate test-backed correction.

## 4. D3.25-C — adapter correction

Replace the union block (`:275-307`) with, after ingress validation:

```python
constant_records, variable_records = normalize_trial_populations(...)
accumulator["bidirectional"].extend(constant_records)
accumulator["bidirectional"].extend(variable_records)
```

This preserves legacy **trial-major, mode-minor** ordering (trial N constant,
then trial N variable) — deterministic, deciding no winner; D3.5's explicit L2
key remains authoritative. `TestResult.bidirectional_count` may still expose
the combined total as run telemetry, but that value must never be copied into a
record's mode-specific `bidirectional_count`.

**Miner path:** D3.25 does NOT rebuild miner records and must not route them
through the new PWC/ZMQ contract. D6 appends the miner's already-canonical
`canonical_records_constant` / `canonical_records_variable` from the stored
`MinerTrialAssembly` without rerunning normalization. Certification status:

```text
PWC/ZMQ both-mode canonical candidate output   uncertified until D3.25
miner  both-mode run-level candidate output    uncertified until D6
```

## 4a. D3 conformance — a free gate that did not exist at REV2

D3 (committed `c207e3a`) added `utils/canonical_arrays.py`, which is now the
**executable definition of a valid canonical 24-field record**: exact 24-key set
(missing OR extra fails), `prng_type`/`prng_base`/`skip_mode` identity
consistency, `prng_base` restricted to a forward non-hybrid base family,
`sessions` as `list[str]`, integer-valued counts, `float32`-representable
finite floats, unit-interval rates and score.

**Binding on D3.25:** every record returned by `normalize_trial_populations`
must pass `utils.canonical_arrays.records_to_arrays` **without raising**. This
is stronger than asserting field values by hand, and it is free — D3 already
paid for it.

Two consequences the normalizer must respect:

- `sessions` must be a real `list[str]`. D3 rejects a tuple, a scalar string
  (`"all"`), and `None` — which is exactly the §3.4 canonical form, now
  machine-checked rather than asserted in prose.
- `skip_range` must be an `int`. D3 rejects the legacy `"5-56"` string form
  outright, so the §6 prohibition is now enforced by the columnizer as well as
  by D3.25's own gate.

**Do NOT import D3's validators into D3.25's production path as a substitute
for D3.25's own validation** — the ingress consistency wall (§2) and the
per-mode derivation (§3.3) remain D3.25's responsibility. D3 conformance is a
*gate-side* check on the normalizer's output, not a replacement for producing
correct records in the first place.

## 5. Gates — `tests/test_s172_phase5_d3_25_candidate_ingress.py`

Independent hand-written oracles throughout: never assert a value by importing
the constant the production code used to compute it.

- **G1 producer contract shape** — all four maps present for constant-only,
  both-mode, and pruned trials, validated at **both** producer egress and
  adapter ingress. Missing field → fail closed at each boundary.
- **G2 same seed, both modes, different rates** — seed 42 constant
  `fwd 0.90 / rev 0.70`; variable `fwd 0.55 / rev 0.95`. Expect **two**
  records: constant `score 0.80`, variable `score 0.75`, each retaining its own
  rates. **Pre-fix adapter must fail.**
- **G3 mode-specific aggregates** — deliberately different population sizes per
  mode; assert every derived field against independent hand calculations per
  mode. Any constant-mode aggregate on a variable record fails.
- **G4 intersection consistency** — mutate a returned bidirectional set to
  disagree with its map intersection → failure **before** accumulator
  mutation, asserted as `len(accumulator) before == len(accumulator) after`
  for **both** constant-set and variable-set corruption.
- **G5 `skip_range`** — integer difference; `"5-56"` fails.
- **G6 `sessions`** — list form, `None → []`; missing attribute and scalar
  `"all"` fail closed. **Additionally:** mutating the caller's original
  sessions list after normalization must not mutate already-produced PWC/ZMQ
  candidate records. (Applies to the PWC/ZMQ wrapper only — not retroactively
  to D1's accepted shared-reference behavior.)
- **G7 no generic-map authority** — supply *misleading* legacy
  `forward_map`/`reverse_map` alongside correct v2 maps; canonical records must
  derive exclusively from the v2 maps.
- **G8 ordering + dual preservation** — one trial with overlapping mode seeds:
  constant ascending, then variable ascending; both survive to L2.
- **G9 canonical oracle (load-bearing)** — since D1 and PWC/ZMQ now call the
  SAME extracted helper, their direct equality is a **regression check, not
  independent proof**. The hand-written 24-field oracle carries the weight and
  must cover: field names and order; all derived formulas; PRNG identity; skip
  mode; integer `skip_range`; canonical sessions form; same-seed preservation
  across modes.
- **G10 record-list-is-not-a-map** — construct a PWC-shaped result whose
  `forward_records_hybrid` repeats a seed while `forward_map_variable` does
  not; assert the normalizer derives from the map and that a
  reconstruction-from-records implementation would differ.
- **G12 D3 conformance (new at REV3)** — every record from
  `normalize_trial_populations`, for a both-mode fixture including a cross-mode
  seed, passes `utils.canonical_arrays.records_to_arrays` without raising; and
  the bundle it produces passes `validate_array_bundle`. Add negative cases: a
  normalizer that emits `sessions` as a tuple, or `skip_range` as the `"5-56"`
  string, must be caught by this gate.
- **G11 mutation proof** — kill each of: restored set union; variable records
  using constant maps; variable aggregates using constant counts; string
  `skip_range`; `"all"` sessions fallback; **removal of adapter-ingress
  validation while producer validation remains intact**; **reconstructing a
  missing map from `forward_records_hybrid`**; **a missing v2 field defaulting
  to `{}`**. Report each red signature.

**Blocking non-regression:** D3.0's gate, D2 7/7, D1.1 18/18, D1.0 8/8, D0
12/12, Phase 4 63/63, Phase 3 17/17. Baseline captured green BEFORE any edit.
D1/D2 green post-extraction is the semantics-preserving proof, and must
include: record key order unchanged; D1 `sessions` ownership unaltered; no new
encoding fallback; no change to `DirectionalDuplicateError`, assembly ordering,
or formulas.

## 6. Scope boundaries

**May modify:** `utils/canonical_records.py` (new),
`miner/range_miner_npz_writer.py`, `persistent_worker_coordinator.py`,
`zmq_sqlite_coordinator.py`, `window_optimizer_integration_final.py`, the new
test, the exact gate-22 whitelist registration, D3.25 governance docs.

**Must NOT modify:** L2 batch winner selection; L3 prior merge; NPZ
columnization; canonical final writing; D3's array contract; D6's miner
adapter; WATCHER behavior. Discovering a necessary change in any of these
triggers **STOP and review**.

## 7. Stop conditions

- extracting `_mode_records` cannot be done without changing semantics;
- a producer cannot construct one of the four maps without recomputation
  (i.e. genuinely absent, not merely unreturned);
- either consistency wall fires on a legitimately-driven real fixture;
- any change is needed in the §6 must-not-modify list;
- any gate passes only by weakening it.

## 8. Report

Diff + status, full command/output evidence, pre-fix failure capture
(G2/G3/G5/G6 red before edits), mutation evidence with signatures, and
confirmation that D1/D2 gates are green post-extraction with the §5 checklist
satisfied. Then STOP for Team Alpha review.
