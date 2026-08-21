# PROPOSAL — WINDOW-ANCHOR / GENERATOR-PHASE SEPARATION — v1.0

**Author:** Team Alpha · **Date:** 2026-08-18 · **Status:** DRAFT for Beta review
**Authority:** `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` (`73633e7`) — proposal phase
authorized; scope (a)+(b) mandatory, (c) OUT; frozen kernel ABI BINDING; `generator_phase=0`
in v1; per-variant capability matrix required; consumer law firewalled.
**Certified pre-change reference:** `gate12-passed-attempt9` = `e9ca800`.
**All line numbers verified at `24ed568` (fresh clone, this session).**

---

## 1. The defect being repaired (F-4), restated precisely

One scalar, `offset`, performs two unrelated jobs in every Step-1 trial:

- **Job A (host, data selection)** — `miner/range_miner_worker.py:649-650`:
  `start = max(0, min(int(offset), n − window_size)); window = data[start : start+window_size]`
  — which observed records form the residue window.
- **Job B (device, generator pre-advance)** — same scalar enters `BuildContext(offset=…)`
  (`:948`), is delivered as `ScalarArg(ctx.offset, "int32")` (`:197-198`), and the kernel
  runs `for (o = 0; o < offset; o++) state = advance(state)` before the skip warm-up
  (`prng_registry.py:974-976`, java_lcg_hybrid; same pattern all families).

The two meanings coincide only at skip = 0. Combined with the optimizer bound
`search_bounds.offset = [0,100]` (`distributed_config.json:80-83`;
`window_optimizer.py:142-143`) and `window_size ≤ 50`, the sieve's reach ceiling is filtered
index **149** — the production system has never examined a draw governed by the CA
procedures document (first governed index: midday **6,791**, evening **7,830**, both
**14,621**; 3,447/18,068 = 19.1% of records governed, none reachable). Chapter 2 F-4:
CONFIRMED, not repaired. D.1 is blocked on exactly this.

## 2. Binding semantic contract (from the ruling, verbatim in effect)

| name | means ONLY | lives |
|---|---|---|
| `window_anchor` | which observed records form the residue window: `window = filtered_data[anchor : anchor+window_size]` | host, residue construction |
| `generator_phase` | how many generator-state advances occur before the first comparison | device, the existing kernel `offset` argument where one exists |

Never reconstructed from one another. Never emulated (no anchor/skip/seed/slice tricks to
fake a phase on variants that lack one). Both recorded independently in every trial record
and artifact.

## 3. Per-variant generator-phase capability matrix (verified at source)

Worker builder functions + ABI header contract, `miner/range_miner_worker.py:23-37`,
builders `:213-360`; kernel signatures cross-checked in `prng_registry.py`.

| variant | args | phase input | v1 `generator_phase` |
|---|---|---|---|
| forward constant, all 6 families | 13 (lcg32: 16) | trailing `int32 offset` | 0, delivered |
| reverse constant, all 6 families | 12 | trailing `int32 offset` (`:210`) | 0, delivered |
| `lcg32_hybrid` forward | 17 | inline `int32 offset` (`:244-249`) | 0, delivered |
| `pcg32_hybrid` forward | 15 | trailing `int32 offset` (`:295-298`) | 0, delivered |
| `java_lcg_hybrid` forward | 15 | **NONE** (`:220`) | 0 / UNSUPPORTED |
| `minstd_hybrid` forward | 15 | **NONE** (`:271`) | 0 / UNSUPPORTED |
| `xorshift32_hybrid` forward | 16 | **NONE** (`:312`) | 0 / UNSUPPORTED |
| `xorshift128_hybrid` forward | 16 | **NONE** (`:35`) | 0 / UNSUPPORTED |
| reverse hybrid, all 6 covered families | 14 | trailing `int32 offset` (`:202-203`) | 0, delivered |

The matrix is encoded as data (a registry table, not scattered `if`s) and enforced at
**registration/validation time**: a trial requesting `generator_phase ≠ 0` on an
UNSUPPORTED variant is rejected fail-loud. In v1 the pin makes this unreachable, but the
enforcement ships now so ABI-v2 cannot arrive before its guard.

## 4. Design

### 4.1 Trial-input schema

`WindowConfig` (`window_optimizer.py:106-127`) and the coordinator assign-payload schema
(8 sites: `range_miner_coordinator.py:1836, :2574, :2588, :2605, :2618, :2642, :2684,
:2751`) replace the single `offset` with:

- `window_anchor: int` — REQUIRED, sampled/searched.
- `generator_phase: int` — REQUIRED, **pinned 0 in v1**, not an Optuna dimension, not
  WATCHER-tunable, not present in `search_bounds`. Carried explicitly (not defaulted) so
  every artifact records the phase that actually ran.

**The legacy fused key `offset` is REJECTED in new-schema payloads** (fail-loud, named
error), not silently mapped — silent mapping would recreate F-4 one abstraction up. The
payload schema version bumps; old-schema payloads are refused by version, giving a clean
mixed-fleet failure mode instead of a semantic one.

### 4.2 Anchor domain — derived, validated, never clamped

Per the ruling, the legal domain is data-derived:
`0 ≤ window_anchor ≤ N_filtered − window_size`, where `N_filtered` is the record count
AFTER the session filter (`:642-644`) — so the domain is session-dependent and resolved
against the dataset content identity already computed by `ResidueResolver` (dataset sha256
cache key, `:659+`).

**The silent clamp at `:649` (`min(int(offset), n − window_size)`) is REMOVED** and replaced
with fail-loud validation: an out-of-domain anchor raises `ResidueResolutionError` naming
anchor, domain, session set, and dataset hash. Rationale: post-separation, a clamped anchor
would silently analyze different draws than the trial record claims — fatal to D.1 and to
any era-labeled result. (`n < window_size` already raises, `:645-647`; unchanged.)

Era subdomains (governed / control) sit on top as **named anchor ranges resolved from the
dataset**, not hardcoded indices: `governed_era = [first_governed_index(session_set),
N_filtered − window_size]`, `control_era = [0, min(149, N_filtered − window_size)]` (the
historical reach, kept as the natural control). D.1 selects by era name; indices are
computed per dataset+session at trial-config build and recorded in the trial context.

### 4.3 Optuna surface

`search_bounds.offset` is **retired** (kept in config as a tombstone comment pointing here).
New: `search_bounds.window_anchor` with `min/max` interpreted against the derived domain —
config may narrow the domain, never widen it; the effective bound is
`min(config.max, N_filtered − window_size)` with a startup log line stating the resolved
domain per session set. Sampling (`:231`) and validation (`:242`) move to the new field.
The cache key (`:127`) becomes `…_A{anchor}_P{phase}_…` so pre- and post-separation trial
identities can never collide.

### 4.4 Worker

- Residue path: `load_residue_window` and `ResidueResolver` take `window_anchor`; cache key
  field renames accordingly (content-identity semantics unchanged, B1 preserved).
- Kernel-arg path: `BuildContext.offset` is renamed `BuildContext.generator_phase`;
  `_offset_tail` / `_reverse_hybrid_tail` deliver it unchanged in position and dtype —
  **kernel ABI byte-for-byte identical**, per the binding constraint. No builder's arg
  count, order, or dtype changes. The four no-phase forward-hybrid builders are annotated
  UNSUPPORTED in the capability table; they receive nothing, as today.
- Validation: capability-matrix check (§3) + anchor-domain check (§4.2) both fail-loud
  before any GPU work.

### 4.5 Artifacts and provenance

NPZ generation metadata (`miner/range_miner_npz_writer.py:188` field tuple) replaces
`offset` with `window_anchor` and `generator_phase`, plus `anchor_era` (string, nullable)
when an era subdomain was used. **Metadata-dict change only — the 22-array contract is
untouched** (the wall binds the array set; Beta is asked to confirm this reading in §8-Q2).
Ledger/trial-record fields follow the same rename at the coordinator sites listed in §4.1.

### 4.6 WATCHER / parameter registry

`agents/registry/parameter_registry.py:282-289` currently exposes `offset` with
`max_value=500` — **stale and wrong twice over** (config says 100; and the number is about
to stop existing). The entry is replaced by `window_anchor` (max documented as
data-derived; registry carries the config-narrowing semantics of §4.3). `generator_phase`
is **not** registered — WATCHER cannot tune it in v1 by construction, not by policy.

### 4.7 Explicitly unchanged (the firewall list)

Kernels and ABI (byte-for-byte, all 44) · skip semantics and bounds ((c) is OUT) ·
thresholds · striping/leases/staging/backpressure/Gate-12 machinery · Phase-5 writer array
contract · **Step-3 consumer law `offset = train_history_len`**
(`full_scoring_worker.py:284-301`, `DAILY3_CONSUMER_CONTRACT_v1.md:182-185`) — a consumer
continuation law, referred to as **`continuation_phase`** in all design narrative from now
on; this proposal does not repeal, parameterize, rename-in-code, or touch it. Any change
there requires its own ruling.

## 5. ABI-v2 dependency record (per the ruling — recorded, NOT proposed)

Independent generator-phase control on `java_lcg_hybrid`, `minstd_hybrid`,
`xorshift32_hybrid`, `xorshift128_hybrid` forward kernels requires a signature change =
new kernel + parity certification cycle. Recorded as **DEP-ABI-V2**, blocked behind: a
demonstrated experimental need (nothing in D.1 needs it — D.1 runs phase=0 everywhere),
and a Beta ruling opening that cycle. Until then, phase on those variants is 0/UNSUPPORTED
and enforcement (§3) makes violation impossible rather than discouraged.

## 6. Comparability caveat (stated so it is ruled, not discovered)

Historical trials sampled `offset ∈ [0,100]`, which pre-advanced generators as a side
effect wherever a phase input existed. Post-separation trials run phase=0. Therefore
pre/post survivor populations are **not comparable run-to-run**, and no cross-epoch
comparison may be used as acceptance or regression evidence. This is F-4's existing damage
(§2.21 already stamps all historical empirical results confounded), now made explicit.

## 7. Acceptance criteria (post-implementation, per the ruling's semantic gate)

1. **Semantic separation tests** — anchor moves ⇒ residue window moves, kernel args
   byte-identical; phase would move ⇒ residue window identical (v1: proven at phase=0 via
   arg-capture, not by running nonzero phase).
2. **Variant-capability tests** — phase≠0 request on each UNSUPPORTED variant rejected
   fail-loud at validation; supported variants deliver the pinned 0 in the correct slot
   (arg-capture per builder, all 44 registry entries exercised).
3. **Domain tests** — out-of-domain anchor fails loud (no clamp path survives: mutation
   evidence that restoring the clamp is detected); era subdomain resolution correct per
   session set against known dataset indices (6,791 / 7,830 / 14,621).
4. **D.1 differential reach demonstration** — one governed-era-anchored and one
   control-era-anchored trial config, identical geometry, both produce runnable assignments
   whose recorded windows land in the correct eras (reach only; scientific run follows
   separately).
5. **No-fused-path proof** — repo-level: no live code path feeds one value to both roles
   (grep + call-graph evidence over the production path); the legacy
   `reverse_sieve_filter.py` disposition per §8-Q3.
6. **Clean-tree / 30/30 fleet parity** re-proof on the changed worker/coordinator surface,
   kernels unchanged by hash.
7. Full existing batteries green (or pre-existing reds identified as such).

## 8. Questions for Beta (the only open items)

- **Q1 — old-key policy:** §4.1 hard-rejects the legacy `offset` key in new-schema
  payloads. Confirm, or require a one-release deprecation shim (Alpha recommends
  hard-reject: a shim is a fused-semantics revival path).
- **Q2 — metadata carry:** confirm that adding `window_anchor` / `generator_phase` /
  `anchor_era` to generation METADATA (not arrays) does not open the 22-array contract
  wall.
- **Q3 — legacy engine disposition:** `reverse_sieve_filter.py:106-114` carries the same
  fused slice (pre-RANGE-MINER engine). Align it, or formally freeze it as
  historical-only with a header stating it predates the separation? (Alpha recommends
  freeze + header; aligning dead code invites drift.)
- **Q4 — control-era definition:** §4.2 proposes `[0, 149]` (the historical reach) as the
  named control era for D.1 symmetry. Confirm or redefine.
- **Q5 — implementation phasing:** Alpha proposes two implementation briefs — (I) worker +
  schema + validation + tests, (II) optimizer/registry/metadata + D.1 reach demo — each
  with its own gate, both against `gate12-passed-attempt9`. Confirm or merge into one.

## 9. What this buys

The first Step-1 geometry that can place the residue window on draws the CA procedures
document actually governs, with generator phase held at a known value — making D.1 a
one-variable experiment: same geometry, governed era vs control era, does the predicted
skip structure appear where the document applies and not where it doesn't. Every prior
empirical result in this project came from `data[0:150]`; this is the change that ends
that.
