# PROPOSAL — WINDOW-ANCHOR / GENERATOR-PHASE SEPARATION — v1.1

**Author:** Team Alpha · **Date:** 2026-08-18 · **Status:** incorporates
`docs/TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md` in full; per its disposition, this revision
closes the design gate and implementation proceeds to Brief I.
**Supersedes:** v1.0 (retained for audit).
**Authority chain:** `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` (`73633e7`) → v1.0 review
ruling. Certified pre-change reference: `gate12-passed-attempt9` = `e9ca800`.
**Changes from v1.0 are exactly Beta's bounded corrections; no new semantics.** Changed
sections: 4.2, 4.3, 4.5, 4.8 (new), 7.1, 7.3, 7.4, 7.5, 8. The Q4 anchor/extent category
error is corrected everywhere.

---

## 1. The defect being repaired (F-4) — unchanged from v1.0

One scalar, `offset`, performs two unrelated jobs in every Step-1 trial: host-side data
selection (`miner/range_miner_worker.py:649-650` — which records form the residue window)
and device-side generator pre-advance (delivered via `BuildContext(offset=…)` `:948` →
`ScalarArg(ctx.offset, "int32")` `:197-198` → `for (o = 0; o < offset; o++) state =
advance(state)`, `prng_registry.py:974-976`). Coherent only at skip = 0. With
`search_bounds.offset = [0,100]` and `window_size ≤ 50`, the sieve's record reach ends at
filtered index 149; first governed record: midday 6,791 / evening 7,830 / both 14,621
(3,447/18,068 = 19.1% governed, none reachable). Chapter 2 F-4: CONFIRMED, not repaired.
D.1 is blocked on exactly this.

## 2. Binding semantic contract — unchanged from v1.0

| name | means ONLY | lives |
|---|---|---|
| `window_anchor` | which observed records form the residue window: `window = filtered_data[anchor : anchor+window_size]` | host, residue construction |
| `generator_phase` | how many generator-state advances occur before the first comparison | device, the existing kernel `offset` argument where one exists |

Never reconstructed from one another. Never emulated. Both recorded independently in every
trial record and artifact.

**Terminology law (from Q4):** an **anchor** is a window *start index*; a **record
envelope** is the *union of records* a set of anchors+windows can reach. The historical
system had anchor ceiling **100** and record-envelope ceiling **149** (= 100 + 50 − 1).
These are different categories and are never interchanged in this design, its configs, its
tests, or its artifacts.

## 3. Per-variant generator-phase capability matrix — unchanged from v1.0

(Verified at source, `miner/range_miner_worker.py:23-37` + builders; `prng_registry.py`
signatures.)

| variant | args | phase input | v1 `generator_phase` |
|---|---|---|---|
| forward constant, all 6 families | 13 (lcg32: 16) | trailing `int32 offset` | 0, delivered |
| reverse constant, all 6 families | 12 | trailing `int32 offset` | 0, delivered |
| `lcg32_hybrid` forward | 17 | inline `int32 offset` | 0, delivered |
| `pcg32_hybrid` forward | 15 | trailing `int32 offset` | 0, delivered |
| `java_lcg_hybrid` forward | 15 | **NONE** | 0 / UNSUPPORTED |
| `minstd_hybrid` forward | 15 | **NONE** | 0 / UNSUPPORTED |
| `xorshift32_hybrid` forward | 16 | **NONE** | 0 / UNSUPPORTED |
| `xorshift128_hybrid` forward | 16 | **NONE** | 0 / UNSUPPORTED |
| reverse hybrid, all 6 covered families | 14 | trailing `int32 offset` | 0, delivered |

Encoded as data, enforced at registration/validation: `generator_phase ≠ 0` on an
UNSUPPORTED variant is rejected fail-loud. The pin makes this unreachable in v1; the guard
ships now so ABI-v2 cannot arrive before it.

## 4. Design

### 4.1 Trial-input schema — unchanged from v1.0, Q1-confirmed

`WindowConfig` and the coordinator assign-payload schema (8 sites:
`range_miner_coordinator.py:1836, :2574, :2588, :2605, :2618, :2642, :2684, :2751`) replace
`offset` with `window_anchor` (REQUIRED, searched) and `generator_phase` (REQUIRED, pinned
0 in v1, not an Optuna dimension, not WATCHER-tunable, carried explicitly so every artifact
records the phase that ran).

**HARD REJECT (Beta-confirmed):** new-schema inputs containing `offset` fail loud before
assignment or GPU work; old schema versions fail by version. No `offset → window_anchor`,
no `offset → generator_phase`, no dual mapping anywhere in the new production path.

### 4.2 Anchor domain — derived, validated, never clamped **(Q4-corrected)**

General legal domain (unchanged): `0 ≤ window_anchor ≤ N_filtered − window_size`, with
`N_filtered` the post-session-filter record count, resolved against the dataset content
identity already computed by `ResidueResolver`. The silent clamp at `:649` is REMOVED;
out-of-domain anchors raise `ResidueResolutionError` naming anchor, effective domain,
session set, and dataset sha256. (`n < window_size` already raises; unchanged.)

Era subdomains are **named ANCHOR ranges** resolved from the dataset per session set at
trial-config build, recorded in the trial context:

- **`control_era` (corrected):** `[0, min(100, N_filtered − window_size)]` — the historical
  **anchor** ceiling. The old figure 149 is the historical **record-envelope** ceiling and
  is never used as an anchor bound. (v1.0's `[0,149]` was the exact anchor/extent category
  error this design eliminates; rejected by Beta and corrected here and in §§7.3, 8.)
- **`governed_era`:** `[first_governed_anchor(session_set), N_filtered − window_size]`,
  where `first_governed_anchor` is the first governed record index for that session filter
  (midday 6,791 / evening 7,830), so every record in the slice is governed.

**D.1 scientific-run constraints (per ruling):** session-scoped — **midday OR evening,
never combined**; window size, skip geometry, thresholds, seed domain, and trial budget
held constant between governed and control arms. Era names select anchor subdomains;
nothing else differs between arms.

### 4.3 Optuna surface and the derived-maximum machine representation **(Q1 + derived-max corrected)**

**`search_bounds.offset` is REMOVED from `distributed_config.json` outright** — no
tombstone, no comment (JSON has none), no retained key. Its retirement is recorded in the
schema-migration note, the changelog, and this proposal — documentation, not live
configuration. `optimal_window_config.json`, trial records, assignment payloads, cache
identities, and survivor provenance emit the new schema only (repo-wide consumer audit:
Brief II, §8-Q5).

**Exact machine representation of the anchor bound:**

```json
"search_bounds": {
  "window_anchor": { "min": 0, "max_cap": null }
}
```

- `min`: `int ≥ 0`.
- `max_cap`: `int | null`. `null` = no configured cap.
- The **derived structural maximum** `derived_max = N_filtered − window_size` is computed at
  trial-config build from the resolved dataset + session set + trial `window_size`, and is
  ALWAYS applied:
  `effective_max = derived_max if max_cap is null else min(max_cap, derived_max)`.
- Validation: `min ≤ effective_max` else fail-loud at study/trial build; a `max_cap` above
  `derived_max` is legal but inert (min() makes widening impossible by construction, not by
  policy). The resolved `[min, effective_max]` and its inputs (`N_filtered`, session set,
  dataset sha256, `window_size`) are logged at study start and recorded per trial.
- Era subdomains (§4.2) intersect this effective domain; empty intersection fails loud.

Sampling and validation move to `window_anchor`; the trial cache key becomes
`…_A{anchor}_P{phase}_…` so pre/post-separation trial identities cannot collide.

### 4.4 Worker — unchanged from v1.0

Residue path takes `window_anchor` (ResidueResolver cache-key field renames; B1
content-identity semantics unchanged). `BuildContext.offset` → `BuildContext.generator_phase`;
`_offset_tail` / `_reverse_hybrid_tail` deliver it in unchanged position and dtype —
**kernel ABI byte-for-byte identical**. Capability check (§3) and domain check (§4.2) both
fail-loud before any GPU work.

### 4.5 Artifacts and provenance **(Q2-corrected)**

NPZ generation metadata replaces `offset` with `window_anchor`, `generator_phase`, and
`anchor_era` — **metadata-dict change only, explicitly versioned; no array added, removed,
reordered, retyped, or reshaped — the 22-array wall stays closed** (Beta-confirmed).

**`anchor_era` is provenance, never authority:** it is DERIVED at trial-config build from
the resolved dataset/session/anchor relationship (or from a validated named-domain request
that resolved successfully), and is recorded ALONGSIDE the facts that justify it — the
actual `window_anchor`, the effective resolved anchor range, the session set, and the
dataset sha256. No consumer may treat the string as proof of era membership; the recorded
anchor + dataset identity are the proof. An arbitrary caller-supplied era string is
rejected at validation.

### 4.6 WATCHER / parameter registry — sharpened per derived-max ruling

`parameter_registry.py:282-289` `offset` entry (stale `max_value=500`) is REPLACED by
`window_anchor` with `min_value=0` and **no static max**: the registry entry declares its
maximum DYNAMIC and delegates to the §4.3 resolver — the registry cannot state a number
that widens the derived domain, again by construction. `generator_phase` is **not
registered**; WATCHER cannot tune it in v1.

### 4.7 Explicitly unchanged (the firewall list) — unchanged from v1.0

Kernels and ABI (byte-for-byte, all 44) · skip semantics and bounds ((c) OUT) · thresholds ·
striping/leases/staging/backpressure/Gate-12 machinery · Phase-5 array contract · Step-3
consumer law `offset = train_history_len` (**`continuation_phase`** in all design
narrative) — not repealed, not parameterized, not renamed in code, not touched; changes
require their own ruling.

### 4.8 Legacy engine closure **(new, per Q3)**

`reverse_sieve_filter.py:106-114` carries the fused slice, and infrastructure documentation
still describes a `reverse_sieve` coordinator job targeting it — deadness may NOT be
assumed. Brief I therefore includes, in order:

1. **Reachability determination at source:** locate every dispatch route in current
   production code (coordinator job types, CLI entry points, imports) that can reach the
   legacy engine.
2. **If a live route exists:** remove or hard-disable it (fail-loud, named error) — the
   engine is NOT aligned to the new semantics (aligning dead code invites drift; Beta
   concurred with freeze-over-retrofit conditional on closure).
3. **Regardless:** add a fail-loud historical-only entry guard to the engine itself, so
   even a future re-wired route cannot silently run fused semantics.
4. **Evidence:** call-graph/reachability proof in the Brief-I report; tested by acceptance
   criterion 5.

The untouched fused implementation then remains as archival code.

## 5. ABI-v2 dependency record — unchanged from v1.0

Independent phase on the four no-phase forward hybrids = **DEP-ABI-V2**: new kernel +
parity certification cycle, blocked behind demonstrated experimental need (D.1 needs
nothing — it runs phase=0 everywhere) and a Beta ruling opening that cycle.

## 6. Comparability caveat — accepted as written, unchanged

Historical `offset` moved both the data window and (on phase-capable kernels) generator
state; post-separation phase-zero populations are not legitimate regression comparators to
historical populations. No cross-epoch comparison may serve as acceptance or regression
evidence.

## 7. Acceptance criteria **(7.1, 7.3, 7.4, 7.5 corrected)**

1. **Semantic separation tests** — anchor moves ⇒ residue window moves with kernel args
   byte-identical. **Phase independence proven actively (per ruling):** an internal
   builder/arg-capture unit test drives a **synthetic nonzero `generator_phase` on a
   supported ABI** (e.g. `lcg32_hybrid` forward) and proves the phase value lands in the
   correct arg slot while the residue window is byte-identical — while the **public v1
   schema remains fail-closed** against nonzero production phase (its own negative test).
   Zero-observed-on-both-paths is not accepted as independence evidence.
2. **Variant-capability tests** — phase≠0 rejected fail-loud on each UNSUPPORTED variant;
   supported variants deliver the pinned value in the correct slot; all 44 registry entries
   exercised via arg-capture.
3. **Domain tests** — out-of-domain anchor fails loud; mutation evidence that restoring the
   clamp is detected; era subdomain resolution per session set against known indices
   (6,791 / 7,830 / 14,621); **control-era resolution asserts the CORRECTED bound
   `[0, min(100, N_filtered − window_size)]`, and a regression test asserts anchor 149 with
   window 50 is NOT inside `control_era`** (the Q4 error, encoded as a permanent test).
4. **D.1 differential reach demonstration** — one governed-era and one control-era trial
   config, **single-session (midday or evening), identical geometry** per §4.2, both
   producing runnable assignments whose recorded windows land in the correct eras.
5. **No-fused-path proof + legacy closure** — repo-level evidence no live path feeds one
   value to both roles; §4.8 executed: route removed/disabled or reachability evidence,
   entry guard present and tested fail-loud.
6. **Clean-tree / 30/30 fleet parity** re-proof on the changed surface; kernels unchanged
   by hash.
7. Full existing batteries green (or pre-existing reds identified as such).

## 8. Resolved rulings (replaces v1.0's open questions — nothing remains open)

| v1.0 question | Beta ruling | landed in |
|---|---|---|
| Q1 old-key policy | HARD REJECT; remove `search_bounds.offset` key outright; new-schema-only artifacts; consumer audit in Brief II | §4.1, §4.3 |
| Q2 metadata | APPROVED; wall closed; `anchor_era` provenance-not-authority | §4.5 |
| Q3 legacy engine | Freeze permitted ONLY with executable closure + evidence | §4.8, AC5 |
| Q4 control era | `[0,149]` REJECTED as anchor range; `control_anchor = [0, min(100, N_filtered − window_size)]`; D.1 session-scoped, arms identical | §2, §4.2, AC3, AC4 |
| Q5 phasing | Two briefs; **Brief II starts from the accepted Brief-I commit**; final report shows sequential lineage + full diff to `e9ca800` | §9 |

## 9. Implementation plan (per Q5 ruling)

- **Brief I** (from `gate12-passed-attempt9` lineage on current main): worker + schema +
  capability validation + legacy-engine closure (§4.8) + semantic tests (AC1-3, AC5-7
  scope).
- **Brief II** (starts from the **accepted Brief-I commit**): optimizer + registry +
  metadata/provenance + repo-wide `offset` consumer audit (incl.
  `optimal_window_config.json` migration) + D.1 reach demonstration (AC4; AC7 re-run).
- Final acceptance report: sequential lineage (e9ca800 → Brief-I commit → Brief-II commit)
  plus the full pre/post diff back to `e9ca800`.

## 10. What this buys — unchanged from v1.0

The first Step-1 geometry that can place the residue window on draws the CA procedures
document governs, with generator phase held at a known value — making D.1 a one-variable,
session-scoped differential experiment. Every prior empirical result came from
`data[0:150]`; this ends that.
