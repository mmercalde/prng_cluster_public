# IMPLEMENTATION BRIEF I — WINDOW-ANCHOR / GENERATOR-PHASE SEPARATION

**Audience:** Claude Code on VM101 (`192.168.3.177`), running as `michael`.
**Author:** Team Alpha. **Date:** 2026-08-20.
**Authority:** `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` — v1.1 APPROVED,
design gate CLOSED, Brief I AUTHORIZED, no v1.2 required.
**Design of record:** `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`
(`1bf49a5`). **v1.0 is audit-only — do not build from it.**
**Certified pre-change reference:** `gate12-passed-attempt9` = `e9ca800`.
**Written against:** HEAD `205ae84c8093e75cbbc0967a857a30ed1c3ce434`.

**Digests of the files this brief modifies, at `205ae84` (verify before you start):**

| file | sha256 (first 16) |
|---|---|
| `miner/range_miner_worker.py` | `043522e96b44855f` |
| `miner/range_miner_coordinator.py` | `53b5ce87c02f46c9` |
| `miner/range_miner_npz_writer.py` | `36e2e34c7ab37a7d` |
| `reverse_sieve_filter.py` | `a68646086734fbdc` |

If any digest differs, **STOP** and report — the tree moved under the brief.

**Michael commits. Team Beta reviews before commit. You do not commit or push.**

---

## §0 — SOURCE-VERIFIED CORRECTIONS TO THE DESIGN OF RECORD — READ FIRST

Three statements in v1.1 do not survive a read of live source at `205ae84`. **Build against
this section, not against the proposal text it corrects.** None of them changes a semantic
decision Beta ruled on; all three are factual and all three would have caused a defect.

### C-1 — §3's forward-constant arg-count column is WRONG

v1.1 §3 states: *"forward constant, all 6 families | 13 (lcg32: 16)"*. Counted from the live
kernel signatures in `prng_registry.py` and cross-checked against the builders in
`miner/range_miner_worker.py:215-353`, the true counts are:

| variant | `_constant_prefix` | family tail | trailing `int32 offset` | **total** | kernel signature |
|---|---|---|---|---|---|
| `pcg32` forward const | 11 | 1 (`increment`) | 1 | **13** | `prng_registry.py:466` |
| `java_lcg` forward const | 11 | 2 (`a`,`c`) | 1 | **14** | `:959` |
| `minstd` forward const | 11 | 2 (`a`,`m_val`) | 1 | **14** | `:1086` |
| `lcg32` forward const | 11 | 3 (`a`,`c`,`m`) | 1 | **15** | `:518` |
| `xorshift32` forward const | 11 | 3 (shifts) | 1 | **15** | `:409` |
| `xorshift128` forward const | 11 | 3 (dummies) | 1 | **15** | `:1219` |

**The capability column of §3 is CORRECT and unchanged** — all six forward-constant kernels
do carry a trailing `int32 offset`, and all six reverse-constant kernels are 12
(`_constant_prefix` + `int32 offset`, params hardcoded in-kernel). Only the *arity* figures
were wrong. The hybrid rows in §3 are correct exactly as written and were re-verified:
java 15 / lcg32 17 / minstd 15 / pcg32 15 / xorshift32 16 / xorshift128 16 forward;
14 for every reverse hybrid.

**Consequence:** G-CAP (§4) asserts arity per variant. Use the table above. A test that
asserts 13 for all six forward-constant variants will red on five of them and would have
been "fixed" by loosening the assertion — which is how a capability matrix stops proving
anything.

### C-2 — the silent clamp exists in FOUR places, not one

v1.1 §4.2 says *"the silent clamp at `:649` is REMOVED"*, implying a single site. Live
source has four instances of the identical expression
`start = max(0, min(int(offset), n - window_size))`:

| # | site | path | in Brief-I scope? |
|---|---|---|---|
| 1 | `miner/range_miner_worker.py:649` | `load_residue_window` — **the miner path, shared authority** | **YES — remove** |
| 2 | `window_optimizer_integration_final.py:266` | `_get_residues_for_config` fallback — PWC/ZMQ path | **NO — firewalled** |
| 3 | `sieve_filter.py:184` | `load_draws_from_daily3` — PWC/ZMQ forward path | **NO — firewalled** |
| 4 | `reverse_sieve_filter.py:114` | `load_draws_from_daily3` — legacy fused engine | **NO — §4.8 closure covers it** |

Site 1 is the only one on the miner path, and it is genuinely shared: the coordinator side
reaches the *same function object* through
`window_optimizer_integration_final._miner_residues_for_config:294-296`, which imports
`miner.range_miner_worker.load_residue_window`. Removing the clamp there changes behaviour
on **both** sides of the assignment simultaneously — that is correct and intended (the
shared-authority invariant documented at `range_miner_worker.py:607-627` depends on it),
but it means the fail-loud path must be exercised from the coordinator side too, not just
the worker. G-DOMAIN-2 covers this.

Sites 2 and 3 belong to the PWC/ZMQ engines, which v1.1 §4.7 does not list in the firewall
by name but which Brief II's repo-wide consumer audit owns. **Do not touch them.** Record
them in the report as a Brief-II carry-forward.

Note also: `load_residue_window` keeps a module-level alias
`_load_window_fresh = load_residue_window` (`:657`). It is the same object; do not create a
second implementation behind it.

### C-3 — §4.1's "8 sites" enumeration is INCOMPLETE, and three of them are invisible to the AST scope proof

v1.1 §4.1 lists eight coordinator sites. Live source has **nineteen** `offset` references in
`miner/range_miner_coordinator.py`, resolving to **nine enclosing definitions plus three
module-level constants**. The full, AST-resolved impact set:

| line(s) | enclosing definition | what it is |
|---|---|---|
| 1388 | `MinerLedger._init_db` | **SQLite DDL column `offset_val`** |
| 1831, 1836 | `MinerLedger.set_trial_context` | INSERT column list + bound value |
| **2574** | *module-level* `_TRIAL_GLOBAL_FIELDS` | tuple |
| **2588** | *module-level* `MANDATORY_MANIFEST_METADATA` | tuple |
| **2605** | *module-level* `_SERVE_CONTEXT_REQUIRED` | tuple |
| 2618 | `_trial_context_row_to_ctx` | maps row `offset_val` → key `"offset"` |
| 2642 | `_canonicalize_trial_context` | **immutability comparison key** |
| 2684 | `build_trial_context_from_serve` | fail-closed projection |
| 2751 | `derive_trial_metadata` | manifest metadata projection |
| 9167, 9266 | `RangeMinerCoordinator.build_stripe_assign_payload` | signature + payload key |
| 9452, 9474, 10412 | `RangeMinerCoordinator.serve_trial` | ctx read + threading |
| 11878, 11900 | `RangeMinerCoordinator._dispatch_pending` | threading |
| 12291, 12420 | `run_trial_miner` | ingress param + context key |

**The three module-level tuples move NO AST digest.** `_def_digests`
(`tests/test_s172_mp1_drain_attribution.py:193-213`) walks only `FunctionDef`,
`AsyncFunctionDef` and `ClassDef` bodies. A module-level tuple change is **invisible to both
scope proofs**. G-TUPLE (§4) exists solely because of this — the scope proof cannot be your
only coverage for those three lines.

**Two further consequences that v1.1 does not name:**

**(a) `_canonicalize_trial_context` is a durable-comparison key.**
`MinerLedger.set_trial_context:1820-1854` computes `new_canon` from a fresh ctx and compares
it against `existing_canon` rebuilt from the persisted row. Changing the field name changes
the canonical JSON string. **A pre-change `trial_context` row will no longer compare equal to
a post-change ctx for the same run_id**, and `set_trial_context` will raise
`MinerMetadataError("conflicting immutable trial context…")`. This is a migration hazard, and
because the ledger is per-run and write-once it only bites on a *resumed* run_id. Handle it
per G-MIGRATE (§4): fail loud with a message that names the schema change, never silently
re-key.

**(b) Phase 5 imports the coordinator's canonicalizer and carries its own copy of the field
tuple.** `miner/range_miner_npz_writer.py:50-53` imports `_canonicalize_trial_context`
directly, and `:187-191` declares `_CONTEXT_FIELDS` with `"offset"` in it. At `:1026`,
`ctx = {k: metas[0][k] for k in _CONTEXT_FIELDS}` is a **required-key** comprehension.
If the coordinator emits `window_anchor` and Phase 5 still asks for `offset`, assembly dies
with `KeyError` on the first real trial.

**Therefore Brief I MUST change `miner/range_miner_npz_writer.py:188`** — one line, the
field name inside `_CONTEXT_FIELDS`. This is *not* the §4.5 NPZ generation-metadata work,
which stays in Brief II: no NPZ array is added, removed, reordered, retyped or reshaped, and
the 22-array wall is untouched. It is a cross-phase consistency tuple that must move in the
same commit or the tree is red between Brief I and Brief II — which AC7 forbids and which
the "Brief II starts from the accepted Brief-I commit" lineage makes unavoidable.

**Flag this scope point explicitly to Beta in the report (§8, item 9).** Alpha is not
deciding it unilaterally; the brief takes the only option that leaves a green tree, and Beta
rules at code review.

---

## §1 — BINDING CONSTRAINTS

Carried from the sequencing ruling, the v1.0 ruling and the v1.1 gate-closure ruling. None
is negotiable inside Brief I.

1. **Kernel ABI frozen byte-for-byte, all 44.** The split lives *above* the ABI. Every
   `ScalarArg` keeps its position and dtype. Any need for independent phase on the four
   no-phase forward hybrids (`java_lcg_hybrid`, `minstd_hybrid`, `xorshift32_hybrid`,
   `xorshift128_hybrid`) is **recorded as DEP-ABI-V2 and NOT built**.
2. **`generator_phase = 0` in v1.** Not an Optuna dimension. Not WATCHER-registered. Carried
   explicitly in every payload and artifact so the phase that ran is recorded, never inferred.
3. **Per-variant capability matrix enforced fail-loud** at registration/validation, using the
   corrected §0 C-1 table.
4. **Fail-loud everywhere the clamp at `range_miner_worker.py:649` used to be silent** —
   including the two `payload.get("offset", 0)` defaults at `:695` and `:875`, which are the
   *same class of silence* and are in scope.
5. **Legacy `offset` key hard-rejected, never mapped.** No `offset → window_anchor`, no
   `offset → generator_phase`, no dual read, no compatibility shim, anywhere on the new path.
6. **Step-3 `offset = train_history_len` (`continuation_phase`) untouched.** Not repealed,
   not parameterized, not renamed in code.
7. **No new `def` in `miner/range_miner_coordinator.py`.** Enforced by two certified gates:
   `test_s172_r1_drain_remedy.py` (`DECLARED_ADDED = set()`, empty and deliberately so) and
   `test_s172_mp1_drain_attribution.py:gate_e2_ast_scope_proof`. Every edit is in-place inside
   an existing definition.
8. **SR-1 applies.** Both `DECLARED_CHANGED` sets need updating, with provenance. See §6.
9. **AC1 requires a synthetic nonzero phase on a supported ABI** via arg-capture, while the
   **public v1 schema stays fail-closed** against nonzero production phase.

---

## §2 — THE CHANGE SET

Work in this order. Each step is independently runnable; do not batch edits across steps.

### 2.1 — `miner/range_miner_worker.py`

**(a) `BuildContext` (`:140-154`).** Rename the field `offset` → `generator_phase`. Position
in the dataclass is irrelevant to the ABI; the field is consumed by name.

**(b) `_offset_tail` (`:197-198`) and `_reverse_hybrid_tail` (`:201-203`).** Rename
`_offset_tail` → `_generator_phase_tail`. Both return
`[ScalarArg(ctx.generator_phase, "int32")]` — **unchanged position, unchanged dtype**. Update
the two inline `ScalarArg(ctx.offset, "int32")` occurrences in `build_lcg32:249` and
`build_pcg32:298` the same way. **These six builder call sites are the entire device-side
surface. Nothing else in the builders changes.**

**(c) Capability matrix — NEW module-level data (`:96` region, beside `SUPPORTED_VARIANTS`).**
Add a declarative table, keyed by concrete variant name, valued by a bool
`accepts_generator_phase`. Derive its contents from §0 C-1, **not** from v1.1 §3. Shape:

```
PHASE_CAPABLE_VARIANTS: frozenset = frozenset({
    # every forward constant  (trailing int32 offset)
    "java_lcg", "lcg32", "minstd", "pcg32", "xorshift32", "xorshift128",
    # every reverse constant  (12 args, trailing int32 offset)
    "java_lcg_reverse", "lcg32_reverse", "minstd_reverse",
    "pcg32_reverse", "xorshift32_reverse", "xorshift128_reverse",
    # the two phase-capable forward hybrids
    "lcg32_hybrid", "pcg32_hybrid",
    # every reverse hybrid  (14 args, trailing int32 offset)
    "java_lcg_hybrid_reverse", "lcg32_hybrid_reverse", "minstd_hybrid_reverse",
    "pcg32_hybrid_reverse", "xorshift32_hybrid_reverse", "xorshift128_hybrid_reverse",
})
```

That is 20 of the 24 covered variants. The four UNSUPPORTED are exactly
`java_lcg_hybrid`, `minstd_hybrid`, `xorshift32_hybrid`, `xorshift128_hybrid`.
**Do not compute this set from a suffix rule** — the whole point is that it is irregular, and
a rule that reproduces it today will silently mis-generalize when ABI-v2 lands.

Also add the expected-arity map from §0 C-1 as data, used by G-CAP.

**(d) Capability enforcement.** A new fail-loud guard, raised **before any GPU work** — put it
next to the existing `resolve_builder` / `_validate_variant` guards so it sits on the same
seam. `generator_phase != 0` on a variant not in `PHASE_CAPABLE_VARIANTS` raises a named
error carrying the variant, the requested phase, and the fact that the kernel has no phase
argument. Reuse `VariantStopCondition` **only if** Beta's existing semantics fit; otherwise a
new dedicated exception is cleaner and the worker has no `def`-count constraint. State which
you chose and why in the report.

**(e) `load_residue_window` (`:602-651`).** Rename the parameter `offset` → `window_anchor`.
Replace `:649`:

```
start = max(0, min(int(offset), n - window_size))
```

with a validated domain and a fail-loud raise. `derived_max = n - window_size` (post-session-
filter, per §4.2). Out of domain raises `ResidueResolutionError` naming, at minimum: the
requested anchor, the effective domain `[0, derived_max]`, the session set, and the dataset
path. The existing `n < window_size` raise at `:646-648` is unchanged. **The `data[start:start
+ window_size]` slice itself does not change** — only how `start` is arrived at.

Note the type hint on `ResidueResolver.__init__`'s `loader` parameter (`:673`) mentions the
positional signature; keep it accurate.

**(f) `ResidueResolver.resolve` (`:686-747`).** Replace `offset = payload.get("offset", 0)`
(`:695`) with **required-key** access to `window_anchor` plus required-key `generator_phase`.
A missing key raises `ResidueResolutionError` — no default. Add the hard reject: a payload
carrying the key `"offset"` fails loud *before* any hashing or loading, with a message that
names the schema change and does **not** offer a mapping.

Update the cache key at `:729-732`. Both branches become
`(…, window_size, canonical_sessions, window_anchor, generator_phase)`. Update the docstring
at `:661-663` and the assignment-contract docstring at `:668-670`, which both enumerate the
old field names.

**(g) Sub-stripe execution path (`:875`).** Same treatment as (f): `offset = payload.get(
"offset", 0)` becomes required-key `generator_phase`, and the `BuildContext` construction at
`:945-950` passes `generator_phase=generator_phase`. Invoke the (d) capability guard here,
against the *resolved concrete variant name* (`family`), before the `cp.cuda.Device` block.

**(h) Module docstring (`:1-44`).** The audited-ABI block names `offset` throughout. Update the
prose to say `generator_phase` where it means the device-side pre-advance, and correct the
forward-constant arity claim per §0 C-1. Leave the kernel-signature transcriptions themselves
accurate to the kernels.

### 2.2 — `miner/range_miner_coordinator.py`

**Nine definitions, three module-level tuples. No new `def`. No removed `def`.**

- `MinerLedger._init_db:1388` — DDL column `offset_val` → `window_anchor_val`, and **add**
  `generator_phase INTEGER`. The table is `CREATE TABLE IF NOT EXISTS`, so an existing
  database file keeps the old shape; see G-MIGRATE.
- `MinerLedger.set_trial_context:1831,1836` — column list and bound values.
- `_TRIAL_GLOBAL_FIELDS:2574` — `"offset"` → `"window_anchor"`, **plus** `"generator_phase"`.
  This is now a 10-field trial-global tuple, so the "11-field canonical trial context"
  language throughout becomes **12-field**. Update every docstring that states the count —
  `_trial_context_row_to_ctx:2611`, `range_miner_npz_writer.py:185-186`, `:980-982`, `:1010`.
  A stale count in a docstring is how the next reader mis-sizes an assertion.
- `MANDATORY_MANIFEST_METADATA:2588` — same two changes.
- `_SERVE_CONTEXT_REQUIRED:2605` — same two changes.
- `_trial_context_row_to_ctx:2618` — `"offset": d["offset_val"]` becomes
  `"window_anchor": d["window_anchor_val"]`, plus `"generator_phase": d["generator_phase"]`.
- `_canonicalize_trial_context:2642` — `int(ctx["offset"])` → `int(ctx["window_anchor"])`,
  plus `int(ctx["generator_phase"])`. **Read C-3(a) before editing this one.**
- `build_trial_context_from_serve:2684` — required-key projection, both fields.
- `derive_trial_metadata:2751` — both fields into the manifest metadata dict.
- `RangeMinerCoordinator.build_stripe_assign_payload:9167,9266` — signature parameter
  `offset: int` → `window_anchor: int`, **add** keyword-only `generator_phase: int` with **no
  default** (same structural argument as the D6 threshold contract: no default means a payload
  that omits it cannot be built). Payload keys `"window_anchor"` and `"generator_phase"`.
  The `"offset"` key is **not emitted**.
- `RangeMinerCoordinator.serve_trial:9452,9474,10412` — ctx reads and threading.
- `RangeMinerCoordinator._dispatch_pending:11878,11900` — threading.
- `run_trial_miner:12291,12420` — parameter `offset: Optional[int] = None` →
  `window_anchor: Optional[int] = None`, **add** `generator_phase: Optional[int] = None`,
  both fail-closed passthrough (the REV4 comment at `:12281-12290` explains exactly why the
  default is `None` and not `0` — preserve that reasoning and extend it to the new field).

`validate_trial_metadata:2771-2790` needs no logic change, but its docstring says
*"(offset 0, skip_min 0, trial_number -1)"*. Update the field name.

### 2.3 — `miner/range_miner_npz_writer.py`

**One line.** `_CONTEXT_FIELDS:188` — `"offset"` → `"window_anchor"`, plus
`"generator_phase"`. Plus the docstring field-count corrections noted in 2.2.

**Nothing else in this file changes.** No array, no dtype, no shape, no `savez` call. The
22-array wall stays closed. Every other `offset` in this file (`seed_offsets`,
`decode_frame(offset=…)`) is a **byte/array offset and is firewalled** — see §3.

### 2.4 — `reverse_sieve_filter.py` — §4.8 legacy-engine closure

**v1.1 §4.8 step 1 says deadness may not be assumed. It is not dead.** Reachability at source,
`205ae84`:

| route | site | kind |
|---|---|---|
| `distributed_worker.py:291` | `analysis_type == 'reverse_sieve'` → `subprocess.run("python3 reverse_sieve_filter.py --job-file …")` | **live dispatch** |
| `coordinator.py:832-838` | `job.search_type == 'reverse_sieve'` → `python -u reverse_sieve_filter.py --job-file …` | **live dispatch** |
| `run_complete_pipeline.py:74-80` | "STEP 3: Reverse Sieve" subprocess | **live dispatch** |
| `identify_failures.py:30`, `identify_failures_trace.py:5`, `test_real_candidates.py:3`, `retest_seed87.py:3` | `from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3` | direct import |

Execute §4.8 in its stated order:

1. **Reachability determination** — reproduce the table above yourself from live source, with
   a scripted call-graph/grep proof committed as evidence. Do not copy this table; verify it.
   If you find a route this brief missed, that is the finding, report it.
2. **Hard-disable the three live dispatch routes.** Fail-loud with a named error at the
   dispatch site — not a silent skip, not a warning-and-continue. The error must state that
   the legacy fused engine is closed under the window-anchor separation and name the design
   doc. Do **not** align the engine to the new semantics; v1.1 is explicit that aligning dead
   code invites drift and Beta concurred with freeze-over-retrofit **conditional on closure**.
3. **Entry guard on the engine itself.** Fail-loud at module entry / `main()` so a future
   re-wired route cannot silently run fused semantics. The four direct-import consumers are
   diagnostic scripts; the guard should fire on the *execution* path, and importing
   `load_draws_from_daily3` for inspection need not explode — state the boundary you chose
   and why.
4. The untouched fused implementation remains as archival code. **Do not delete it.**

---

## §3 — FIREWALL: WHAT DOES NOT CHANGE

**The single most dangerous thing you could do in this brief is a textual rename.** The token
`offset` carries at least four unrelated meanings in this repo. Rename by **semantic
identity**, never by string match. A repo-wide `sed s/offset/window_anchor/` would corrupt the
protocol framer, the checkpoint encoder and the NPZ seed blob.

**Firewalled — these are byte/array offsets and have nothing to do with the window:**

- `miner/range_miner_protocol.py:242-271` — `decode_frame(data, offset)`,
  `message_from_bytes(data, offset)`, `next_offset`. Frame parsing.
- `utils/checkpoint_d6_2.py` — `sessions_offsets`, `_SESSIONS_OFFSETS`, the CSR encoder.
- `miner/range_miner_npz_writer.py:452-557` — `seed_offsets`, the uint64 seed-blob index.

**Firewalled — different subsystem, Brief II or out of scope entirely:**

- **Step-3 `offset = train_history_len` / `continuation_phase`** — constraint 6. Untouched.
- `agents/registry/parameter_registry.py:281-289` — the `offset` `ParameterSpec` with stale
  `max_value=500`. **v1.1 §4.6 assigns this to Brief II** (registry). Leave it. Record it.
- `window_optimizer.py`, `window_optimizer_bayesian.py`,
  `window_optimizer_integration_final.py` — the Optuna surface, `SearchBounds`,
  `WindowConfig`, `suggest_int('offset', …)`, the `O{offset}` cache-key fragment at
  `:2354`, `optimal_window_config.json`. **All Brief II.**
- `distributed_config.json` `search_bounds.offset`. **Brief II.** See the C-4 note below.
- `sieve_filter.py`, PWC and ZMQ residue paths, `zmq_sqlite_coordinator.py`. **Brief II
  consumer audit.**
- Kernels, skip semantics and bounds, thresholds, striping/leases/staging/backpressure/
  Gate-12 machinery, the Phase-5 array contract. **§4.7 firewall list, unchanged.**

### C-4 — a Brief-II hazard found during this read, recorded so it is not lost

v1.1 §4.3 says `search_bounds.offset` is **removed from `distributed_config.json` outright —
no tombstone, no retained key.** At source, `window_optimizer.py:67-92`
(`load_search_bounds_from_config`) **merges the config over a hardcoded defaults dict** which
contains `"offset": {"min": 0, "max": 100}` at `:74`. Deleting the JSON key therefore does
**not** retire the bound — it silently falls back to the hardcoded default, and
`SearchBounds.from_config:164-166` reads it with required-key access into `min_offset` /
`max_offset`, which then feed `suggest_int('offset', …)`.

**Brief II must delete the defaults entry at `window_optimizer.py:74` as well**, or §4.3's
"removed outright" is not true in the running system. Not Brief-I work. Put it in the report.

*(The live config's `offset.max` is 100, which is consistent with §2's terminology law:
100 is the anchor ceiling; 149 is the record-envelope ceiling. Do not write `[0,149]`
anywhere as an anchor bound — see self-check #19 and G-ENVELOPE.)*

---

## §4 — GATES

Every gate names **the wrong input that reds it**. A gate whose failure mode you cannot state
is not a gate. All gates live under `tests/`; follow the existing naming and harness
conventions in `tests/test_s172_phase3_worker.py` and `tests/test_s172_phase4_coordinator.py`.

### Capability & ABI

**G-CAP-1 — arity per variant.** Arg-capture (no GPU) across **all 24 covered variants**,
asserting the exact arity from §0 C-1 and the exact `(position, dtype)` of every scalar.
*Reds on:* any builder that changes an arg position, a dtype, or a count. Assert the numbers
as literals from the C-1 table — do not compute them from the builder you are testing.

**G-CAP-2 — the 20 remaining registry entries.** `KERNEL_REGISTRY` has 44 entries;
`SUPPORTED_VARIANTS` covers 24 (6 covered families × 4). AC2's "all 44 exercised" is
satisfied for the other 20 by asserting they raise — `NotImplementedError` from
`resolve_builder:366-378` (uncovered base family) or `VariantStopCondition` from
`_validate_variant:423-440`. *Reds on:* a variant silently acquiring a builder, or a covered
variant regressing into the uncovered set.

**G-CAP-3 — phase rejection on the four UNSUPPORTED variants.** `generator_phase != 0` on
`java_lcg_hybrid`, `minstd_hybrid`, `xorshift32_hybrid`, `xorshift128_hybrid` fails loud,
**before any GPU work**, with the variant named in the message. *Reds on:* a guard placed
after device acquisition, or one that clamps to 0 instead of raising.

**G-CAP-4 — pinned delivery on the 20 supported variants.** `generator_phase = 0` lands in
the correct arg slot, correct dtype. *Reds on:* the phase being dropped, or delivered in the
wrong position.

**G-ABI-FROZEN — kernel hash equality.** Hash every `kernel_source` string in
`KERNEL_REGISTRY` pre- and post-change; assert all 44 identical. *Reds on:* any kernel edit
whatsoever. This is AC6's "kernels unchanged by hash".

### Semantic separation — AC1

**G-SEP-1 — anchor moves, kernel args do not.** Two residue resolutions differing only in
`window_anchor` produce different residue windows while the captured kernel arg tuple is
**byte-identical** apart from the residue buffer contents. *Reds on:* any re-coupling where
the anchor reaches a scalar arg.

**G-SEP-2 — synthetic nonzero phase on a supported ABI (AC1, mandatory).** Drive
`lcg32_hybrid` forward through the **internal builder** with `generator_phase = 7` via
arg-capture. Assert: (i) 7 appears at the correct trailing `int32` position; (ii) the residue
window is byte-identical to the `generator_phase = 0` case. **Zero-observed-on-both-paths is
explicitly not accepted as independence evidence** — the ruling requires an active nonzero
drive. *Reds on:* a test that only ever observes 0, or one that reaches the kernel through
the public schema (which must reject nonzero — that is G-SEP-3, a different test).

**G-SEP-3 — the public v1 schema stays fail-closed.** A production-shaped assignment payload
with `generator_phase != 0` is rejected fail-loud at validation. *Reds on:* the pin being
relaxed, or nonzero leaking through the assignment path. **G-SEP-2 and G-SEP-3 must both be
green simultaneously** — that pairing is the whole of AC1.

### Domain — AC3

**G-DOMAIN-1 — out-of-domain anchor fails loud (worker side).** `window_anchor > n -
window_size` raises `ResidueResolutionError` naming anchor, effective domain, session set and
dataset. *Reds on:* a clamp, a silent min(), or a bare exception with no diagnostic payload.

**G-DOMAIN-2 — same failure from the coordinator side.** Drive
`window_optimizer_integration_final._miner_residues_for_config` with an out-of-domain anchor
and assert the identical raise. This exists because C-2 established that both sides call the
same function object; a fix that only fails loud when reached from the worker has not
actually removed the silent path.

**G-DOMAIN-3 — session-scoped derived max.** `derived_max` is computed on the
**post-session-filter** count. For `sessions=['midday']` vs `['midday','evening']` on the same
dataset, the effective domains differ. *Reds on:* `derived_max` computed pre-filter — which
would let a midday-only trial address an anchor past the end of its own filtered sequence.

**G-DOMAIN-4 — the `n < window_size` raise is preserved.** *Reds on:* the new validation
swallowing or reordering the existing short-dataset error.

**G-ENVELOPE — the permanent Q4 regression test (AC3, mandatory).** Assert that
`control_era`'s anchor bound is `[0, min(100, N_filtered − window_size)]`, and — explicitly —
that **anchor 149 with window 50 is NOT inside `control_era`.** Name it so its purpose is
unmissable. *Reds on:* anyone reintroducing 149 as an anchor ceiling. This is the exact
category error Alpha committed inside the design written to eliminate it; it is encoded as a
test so it cannot recur silently.

*(Era-subdomain resolution against the governed indices 6,791 / 7,830 / 14,621 is AC3's
remaining clause. If era resolution lands in Brief II with the optimizer surface, state that
explicitly and scope G-ENVELOPE to the bound arithmetic alone — do not quietly drop the
assertion.)*

### Schema rejection — §4.1 HARD REJECT

**G-REJECT-1 — legacy `offset` key.** A payload containing `"offset"` fails loud before any
hashing, loading, assignment or GPU work. *Reds on:* the key being ignored, mapped, or
reaching a later validation stage.

**G-REJECT-2 — no mapping anywhere.** Assert the string `"offset"` does not appear as a dict
key read or written on the new production path. *Reds on:* a compatibility shim added under
time pressure.

**G-REJECT-3 — missing key is not a default.** A payload omitting `window_anchor` or
`generator_phase` raises. *Reds on:* a `.get(…, 0)` surviving anywhere on the path — this is
the `:695` / `:875` defect class and it must be extinct.

### Ledger & cross-phase

**G-TUPLE — module-level tuple contents.** Assert the exact contents of
`_TRIAL_GLOBAL_FIELDS`, `MANDATORY_MANIFEST_METADATA`, `_SERVE_CONTEXT_REQUIRED` and
`range_miner_npz_writer._CONTEXT_FIELDS`. **This gate exists because the AST scope proof
cannot see module-level constants** (C-3). *Reds on:* a tuple edited without the others, or
a field added to one and forgotten in the rest.

**G-PHASE5-SEAM — coordinator and Phase 5 agree.** Build a manifest through
`derive_trial_metadata`, then run it through `prepare_trial_assembly`'s `_CONTEXT_FIELDS`
comprehension and `_canonicalize_trial_context`. *Reds on:* the `KeyError` at
`range_miner_npz_writer.py:1026` that C-3(b) predicts if the two tuples drift.

**G-MIGRATE — legacy ledger row.** Point the coordinator at a `trial_context` table written
with the old `offset_val` schema and assert a **loud, named** failure that identifies the
schema change. *Reds on:* a silent `KeyError` traceback, a fabricated default, or — worst —
`INSERT OR IGNORE` succeeding against the old table and the canonical comparison then failing
with the generic "conflicting immutable trial context" message, which would send the next
reader hunting a phantom config conflict.

### Legacy closure — AC5

**G-LEGACY-1 — reachability proof.** The scripted call-graph/grep evidence runs and produces
the route table. *Reds on:* a new route appearing.

**G-LEGACY-2 — dispatch routes closed.** Each of the three live routes fails loud when
exercised. *Reds on:* a route silently skipping instead of raising.

**G-LEGACY-3 — engine entry guard.** The guard fires on the execution path. *Reds on:* the
engine running fused semantics through any path.

**G-NO-FUSED — AC5's repo-level proof.** No live path feeds one value to both the residue
slice and the generator pre-advance. *Reds on:* a single scalar reaching both roles.

### Scope

**G-SCOPE-COORD — AST scope proof, both sets.** Both existing gates
(`test_s172_r1_drain_remedy.gate_scope_proof`, `test_s172_mp1_drain_attribution.
gate_e2_ast_scope_proof`) green **after** the SR-1 updates in §6. *Reds on:* any new or
removed `def` in the coordinator, or any changed def not declared in **both** sets.

**G-BATTERY — AC7.** Full existing test batteries green, or every pre-existing red identified
as pre-existing **with evidence from `205ae84` before your changes**. Capture that baseline
**first**, before you edit anything. A red you cannot prove was already red is your red.

---

## §5 — MUTATION EVIDENCE

Per the mutation-evidence rule: **prove each mutant actually applied, executed the mutated
path, and reached the credited assertion.** A mutant that dies on import, on class identity,
on a loader error, or on a `TypeError` before reaching the mutated line **earns no credit** —
report it as an invalid mutant and replace it. For each, record: the diff applied, evidence
the mutated line executed, the gate that caught it, and the assertion text.

| # | mutation | must be caught by |
|---|---|---|
| M1 | Restore the clamp: `start = max(0, min(int(window_anchor), n - window_size))` | G-DOMAIN-1 **and** G-DOMAIN-2 |
| M2 | Reinstate `payload.get("window_anchor", 0)` | G-REJECT-3 |
| M3 | Accept and map `offset` → `window_anchor` | G-REJECT-1, G-REJECT-2 |
| M4 | Add `java_lcg_hybrid` to `PHASE_CAPABLE_VARIANTS` | G-CAP-3 |
| M5 | Drop the phase arg from `_generator_phase_tail` | G-CAP-1, G-CAP-4 |
| M6 | Move the phase arg one position earlier in `build_lcg32` | G-CAP-1 |
| M7 | Compute `derived_max` **before** the session filter | G-DOMAIN-3 |
| M8 | Set `control_era` ceiling to 149 | G-ENVELOPE |
| M9 | Revert `_CONTEXT_FIELDS` in the NPZ writer to `"offset"` | G-TUPLE, G-PHASE5-SEAM |
| M10 | Re-enable one legacy dispatch route | G-LEGACY-2 |
| M11 | Change one byte of one `kernel_source` | G-ABI-FROZEN |
| M12 | Add a new `def` to `range_miner_coordinator.py` | G-SCOPE-COORD (both sets) |
| M13 | Let `generator_phase = 3` through the public schema | G-SEP-3 |
| M14 | Make the anchor reach a kernel scalar arg | G-SEP-1, G-NO-FUSED |

M1 is deliberately listed against two gates: AC3 requires "mutation evidence that restoring
the clamp is detected", and C-2 established the shared-authority path means one-sided
detection is insufficient.

---

## §6 — SR-1 OBLIGATIONS

Standing rule, no ruling request needed. Any authorized commit touching a module under a
historical exact live-vs-anchor scope gate **must update every affected `DECLARED_CHANGED`
set before the commit is accepted**. Four constraints: the anchor never moves;
`changed == DECLARED_CHANGED` is never relaxed to subset/superset; only definitions actually
changed are added; **every added entry carries provenance naming the change that owns it**.

Both sets take the same nine definitions. Follow the existing in-file comment style — the
FIELD-6 entries at `test_s172_r1_drain_remedy.py:2129-2139` and
`test_s172_mp1_drain_attribution.py:1415-1421` are the pattern.

**Add to `test_s172_r1_drain_remedy.py:DECLARED_CHANGED`** (currently 3 entries) and to
**`test_s172_mp1_drain_attribution.py:DECLARED_CHANGED`** (currently 12 entries), each
carrying provenance `[WINDOW-ANCHOR BRIEF I]` and a one-line note that this is **not** R-1's
or MP-1's change:

```
MinerLedger._init_db
MinerLedger.set_trial_context
_trial_context_row_to_ctx
_canonicalize_trial_context
build_trial_context_from_serve
derive_trial_metadata
RangeMinerCoordinator.build_stripe_assign_payload
RangeMinerCoordinator._dispatch_pending
run_trial_miner
```

`RangeMinerCoordinator.serve_trial` is **already** in MP-1's set (from the FIELD-6 pass) but
is **not** in R-1's — add it to R-1's only, with the same provenance.

**`DECLARED_ADDED` stays empty in both.** It is empty deliberately: MP-1's certified gate
asserts the added set exactly, so any new coordinator `def` — however well named — reds a
certified gate. Constraint 7.

**Do not add a definition you did not change.** Run the scope proof and let it tell you the
true changed set; if it disagrees with the nine above, **the scope proof is right and this
brief is wrong** — report the delta rather than editing the declaration to match the brief.

---

## §7 — EXECUTION ON VM101

Host context on every command block. `torch` and `tf` are shell **functions** in `.bashrc`
(interactive only) — a non-interactive SSH session must activate the venv explicitly.

```bash
# run on VM101
source ~/venvs/torch/bin/activate
cd /home/michael/distributed_prng_analysis
git rev-parse HEAD          # expect 205ae84c8093e75cbbc0967a857a30ed1c3ce434
sha256sum miner/range_miner_worker.py miner/range_miner_coordinator.py \
          miner/range_miner_npz_writer.py reverse_sieve_filter.py
```

Verify against the digest table at the top of this brief before editing anything.

**Capture the AC7 baseline BEFORE your first edit** — the full battery result at `205ae84`,
saved to a file. Without it you cannot distinguish a pre-existing red from one you caused.

Then, in order: §2.1 worker → §2.2 coordinator → §2.3 npz writer → §2.4 legacy closure →
§4 gates → §5 mutations → §6 SR-1. Run the batteries between steps, not only at the end.

**You do not commit and you do not push.** `git commit` and `git push` are on the VM101
deny-list. Michael commits after Beta's review.

Write `docs/SESSION_CHANGELOG_YYYYMMDD_WINDOW_ANCHOR_BRIEF_I.md` — **date plus topic, no
S-number** (SR-2, binding).

---

## §8 — REPORT REQUIREMENTS

The report is the artifact Beta reviews. It must contain:

1. Digest verification at start, and the AC7 baseline captured before any edit.
2. The full changed-definition list, **computed** by the scope proof, not transcribed.
3. Both `DECLARED_CHANGED` sets after update, with provenance per entry, and both scope
   proofs green.
4. G-CAP-1 output: the arity and scalar-position table for all 24 covered variants, measured.
5. G-ABI-FROZEN: 44/44 kernel hashes identical.
6. AC1 evidence: G-SEP-2 (synthetic nonzero, supported ABI, arg-capture) **and** G-SEP-3
   (public schema fail-closed) green together, with the captured arg tuples shown.
7. §4.8 reachability proof — the route table you derived, not the one in this brief, plus the
   closure evidence for each route and the entry guard.
8. Mutation table: 14 mutants, each with proof it applied, executed the mutated path, and
   reached the credited assertion. Invalid mutants reported as invalid, not silently replaced.
9. **The three §0 corrections, restated as findings against v1.1**, plus the C-3(b)
   scope point (`range_miner_npz_writer.py:188` pulled into Brief I to keep the tree green)
   flagged for Beta's explicit ruling at code review, and the C-4 hazard
   (`window_optimizer.py:74` hardcoded offset default) recorded as a Brief-II carry-forward
   with the C-2 sites 2 and 3.
10. DEP-ABI-V2 restated as recorded-not-built.
11. Explicit statement that Step-3 `continuation_phase` was not touched, with evidence.
12. AC7: full batteries green, or pre-existing reds identified against the captured baseline.

**Target: this brief closes in ≤3 Beta review rounds.** D6.2 took five and 6-P2 took four, and
every round was an Alpha defect, not reviewer padding. The import gate closed in one — because
the existing gate was read in full before the brief was written. Read first, then draft.

---

## §9 — WHAT THIS BRIEF DOES NOT DO

Brief II, starting from the **accepted Brief-I commit**: the Optuna surface
(`suggest_int('offset')` → `window_anchor`, the `{min, max_cap}` representation and the
derived-max resolver), `parameter_registry.py`, `distributed_config.json` key removal **plus
the `window_optimizer.py:74` defaults entry (C-4)**, NPZ generation metadata and `anchor_era`
provenance (§4.5), `optimal_window_config.json` migration, the repo-wide `offset` consumer
audit including C-2 sites 2 and 3, and the D.1 differential reach demonstration (AC4, AC7
re-run).

Final acceptance report shows sequential lineage `e9ca800 → Brief-I commit → Brief-II commit`
plus the full pre/post diff back to `e9ca800`.

**Not acceptance evidence for any of this:** the Phase-7 soak. It is classified
**non-certifying for window-anchor semantics**. Do not cite it.
