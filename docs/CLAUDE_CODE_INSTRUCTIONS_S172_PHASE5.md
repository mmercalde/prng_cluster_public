# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5.md

**S172 RANGE-MINER — Phase 5 (NPZ writer + assembly) + Phase-5-prerequisite Phase-4 seam correction**

**Audience:** Claude Code, running on VM 101 (`michael@192.168.3.177`), inside
`~/distributed_prng_analysis`. You write and iterate the implementation and its
harness here. You do **not** commit, push, or run WATCHER. When each deliverable's
gate is green you STOP and report; Team Alpha reviews the actual files against
live source, Team Beta reviews, Michael commits + dual-pushes.

**Frozen against repo HEAD `0c3166a`.** Spec authority:
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (§3.B, §6.7.A/B/C, §12.1, §15, §16, §17)
and the frozen `v1_4_4` §4.2 (22-array schema), §4.3 (dedup), §6.8 (4-phase
workflow). Three binding Team Beta rulings absorbed below are marked **[TB-R1]**,
**[TB-R2]**, **[TB-R3]**.

---

## 0. Non-negotiable working rules (read before every deliverable)

1. **Read live source before every claim.** Especially at component seams. Phase 4
   took 7 review rounds because early work extrapolated instead of reading. Open
   the actual file at the actual line before you assert what it does.
2. **Each gate must FAIL on absent or wrong behavior.** A gate that passes against
   a stub proves nothing. Construct the adversarial case, not the happy path.
3. **No test-only shortcuts.** Exercise the real lifecycle: real staged spool files
   on disk, real `Phase5Sink` calls from the real coordinator publish surface, real
   `spawn`/`forkserver` child processes. Do not monkeypatch away the thing under test.
4. **STOP and report at each gate.** Do not chain deliverables. Do not commit, push,
   or invoke `agents/watcher_agent.py --run-pipeline`. Those are deny-ruled on 101
   anyway; do not attempt to work around them.
5. **Do not invent field semantics.** Every one of the 22 NPZ arrays already has a
   defined meaning in existing Step-1 code. Phase 5 *reproduces* those meanings
   exactly. It does not approximate missing populations or invent replacement
   formulas. If you find yourself needing a new formula, STOP and report — that is
   a proposal amendment, not an implementation decision.
6. **`utils/prng_encoding.py` is the single source of truth** for PRNG/skip
   categorical encoding. Do **not** copy the stale 12-key dict in
   `convert_survivors_to_binary.py:31` or the inline `_PRNG_ENC` in
   `window_optimizer_integration_final.py:1703`. The canonical module derives IDs
   from `KERNEL_REGISTRY` and **hard-fails** on unknown families — that hard-fail is
   required behavior, not an inconvenience.

---

## 1. What the producer path already gives you (verified trace, HEAD 0c3166a)

Do not re-derive this; it is the frozen input contract. But verify each cite before
you depend on it.

**The four workflow passes** (`v1_4_4` §6.8, resolver
`range_miner_coordinator.py:workflow_stages_for`, base-parameterized — NEVER
hardcode Java LCG):

| Phase | family_name             | direction | skip_mode |
| ----- | ----------------------- | --------- | --------- |
| 1     | `<base>`                | forward   | constant  |
| 2     | `<base>_reverse`        | reverse   | constant  |
| 3     | `<base>_hybrid`         | forward   | variable  |
| 4     | `<base>_hybrid_reverse` | reverse   | variable  |

Pairing for bidirectional enrichment: **(P1,P2) → constant** population,
**(P3,P4) → variable** population.

**The staged shard** is the complete, hash-verified, threshold-passing directional
survivor population for one sub-stripe. Phase 4 has already transferred it to Zeus
staging and verified its bytes against the worker SHA-256 (`§15`,
`_finalize_stage`). Its on-disk format is the worker's canonical payload
(`range_miner_worker.build_substripe_payload_bytes`), compact sorted-key JSON:

```json
{"schema_version":"s172_substripe_v1","stripe_id":"...","sub_index":N,
 "seed_start":S,"seed_count":C,"survivors":[[seed,match_rate,strategy_id_or_null,[skip...]], ...]}
```

Per-survivor tuple: `(seed:int, match_rate:float, strategy_id:int|null, skip_sequence:list[int])`.
- constant passes emit `(seed, rate, null, [best_skip])`;
- hybrid passes emit `(seed, rate, strategy_id, skip_sequence)`.

**`strategy_id` and `skip_sequence` are NOT part of the frozen 22-array public
schema.** Do not turn them into new NPZ arrays. They may be retained in an
*internal diagnostic* artifact only. [frozen schema: `v1_4_4` §4.2]

**The manifest** (`range_miner_coordinator._build_manifest`) is what Phase 5
receives via `Phase5Sink.publish_shard`. It carries the staged path + size + sha +
`workflow_phase` + `trial_metadata`. **Today `trial_metadata` is always `{}`** —
Deliverable D0 fixes that.

**`serve_trial` returns** `{run_id, state, committed, workers_registered, stripes,
manifests, bound_addr}`. It returns **no** survivor maps. Assembling
`forward_map`/`reverse_map`/`bidirectional` sets from the staged shards is Phase 5's
job — that is the missing work, not lost data.

**Ownership model — the Phase 5 sink owns the canonical assembly result [TB-R2].**
The `"manifests"` list in the `serve_trial` return is coordinator **telemetry/
state**, NOT the adapter's input contract. Do **not** treat it as the thing
`_build_test_result_from_miner` reads. The correct dataflow:

```
publish_shard(manifest)  ← Phase 4 streams each verified shard to the sink as it
                           finalizes; the sink ACCUMULATES manifests internally,
                           keyed by run_id.
commit_trial(event)      ← "all stripes reconciled." event carries only
                           {event_type, run_id, event_id} — NO survivor data. The
                           sink ALREADY holds every published manifest, so THIS is
                           where the sink runs the canonical assembly (D1 engine)
                           and stores the resulting MinerTrialAssembly keyed by
                           run_id. (verify: coordinator.commit_trial ~2550-2596;
                           Phase5Sink ~1180-1200)
abort_trial(event)       ← SYNCHRONOUS (L7). The sink drops its own accumulated
                           manifests + any partial assembly for that run_id and
                           returns only after that is complete. Because the sink
                           owns partial assembly, "discard all provisional shards +
                           partial assembly" is simply the sink clearing its own
                           state — coherent with L7.
```

The integration adapter (`_build_test_result_from_miner`, D6) then **reads the
sink's already-computed `MinerTrialAssembly` for that `run_id`** and converts it.
It does **not** reopen manifests or re-run assembly. One assembly, in the sink;
one adapter that consumes it. Reopening every manifest in the adapter to rebuild
the four maps is the exact double-assembly **[TB-R2]** forbids.

---

## 2. Deliverables (staged; one acceptance gate each; STOP after each)

Order is mandatory. D0 is a prerequisite Phase-4 correction and ships/【reviews】
first because everything downstream depends on the manifest carrying real
identity.

---

### D0 — Phase-4 metadata-seam + durable-context correction  **[TB-R3, TB-R1 seam]**

**Why:** The manifest's `trial_metadata` is always `{}` because `publish_attempt`
calls `_build_manifest` with no metadata argument
(`range_miner_coordinator.py:1586-1588`). Phase 5 cannot populate `trial_number`,
`window_size`, `offset`, `skip_min/max`, `prng_base`, direction, or mode without it.
This is the same class of narrow, seam-local Phase-4 correction as the Phase-3
`ResidueResolver` fix — it does **not** reopen the Phase 4 architecture.

**Scope (one controlled change):**

1. **Real call-site propagation** in
   `window_optimizer_integration_final.py` `_use_miner` gate. Add the currently
   missing `skip_min`/`skip_max` (the site passes `window_size`/`offset`/`sessions`
   but omits skip bounds — verify at lines ~399-402). All values come from the
   resolved `WindowConfig` (`window_optimizer.py:74` `class WindowConfig` — it has
   `window_size, offset, sessions, skip_min, skip_max, forward_threshold,
   reverse_threshold`), never from `run_trial_miner` defaults.

2. **Persist trial-global immutable context ONCE per `run_id`**, before any stripe
   work, in the ledger (extend the `trials` table or an adjacent immutable table —
   your choice, but it must survive restart; do not stash only in the in-memory
   `context` dict). Fields:
   ```
   trial_number, window_size, offset, sessions, skip_min, skip_max,
   prng_base, forward_threshold, reverse_threshold,
   dataset_sha256, residue_sha256
   ```
   `dataset_sha256` is already computed once (`compute_dataset_sha256`, used at
   `serve_trial` ~2742); `residue_sha256 = sha256_residues(residues)` (~2680).
   Copy them — do not recompute.

3. **Phase/stripe-specific context**, persisted or deterministically derived from
   the ledger's existing `stripes.phase` + `stripes.family_name` (both already
   stored — verify schema at ~427-428):
   ```
   workflow_phase, family_name, prng_type, direction, skip_mode, threshold_used
   ```
   Direction/mode derive from the §6.8 table above via the resolved base — express
   it through the workflow-stage resolver, **not** hardcoded to Java LCG.

4. **`publish_attempt` populates every manifest** by threading a complete,
   immutable `trial_metadata` projection into `_build_manifest`. Trial-global
   fields identical across all of a run's manifests; phase-specific fields correct
   per stripe. Also retain `dataset_sha256`/`residue_sha256` as provenance
   (non-NPZ) fields on the manifest.

5. **Fail closed on missing mandatory metadata** *before* publication — a manifest
   that would publish with an absent mandatory field raises, does not emit `{}`.

**Mandatory manifest metadata (minimum):**
```
trial_number, window_size, offset, sessions, skip_min, skip_max,
prng_base, prng_type, family_name, direction, skip_mode, workflow_phase,
forward_threshold, reverse_threshold
```
plus provenance: `dataset_sha256, residue_sha256`.

**Gate D0 — seven checks, each must fail on the wrong behavior [TB-R3]:**
1. every published shard carries all mandatory metadata fields (non-empty);
2. metadata is *identical* where trial-global, *correct* where phase-specific
   (assert P1 manifest says forward/constant, P2 reverse/constant, P3
   forward/variable, P4 reverse/variable, with matching `family_name`);
3. forward/reverse and constant/variable identities are explicit strings, not
   inferred from the numeric phase by the consumer;
4. metadata cannot change after trial creation (mutate the source config
   post-creation → published manifests are unaffected → proves immutability;
   include a restart-recovery reconstruction producing an identical manifest);
5. retry attempt 1 carries the same *semantic* context as attempt 0 (same
   phase/family/direction/mode/thresholds);
6. a manifest missing any mandatory field fails closed *before* Phase 5
   publication (no `{}` leak);
7. commit/abort and acknowledgement behavior is **unchanged** (regression-assert
   the existing terminal-exclusivity + ack paths).

**Non-regression (D0 blocks on these):** existing Phase 4 **63/63**, Phase 3
**17/17**.

---

### D1 — Shared four-population assembly engine (`miner/range_miner_npz_writer.py`)  **[TB-R1, TB-R2]**

The single authoritative derivation engine. Everything else (both backends,
the adapter, Phase 6 equality, the accumulator update) consumes **this one
result** — no parallel assembly implementations. **[TB-R2: "one authoritative
derivation result."]**

**Ownership:** this engine runs **inside the `Phase5Sink`**, driven by
`commit_trial(event)` (see §1 ownership model). The sink accumulates manifests via
`publish_shard`, and on `commit_trial` assembles them into ONE
`MinerTrialAssembly`, stored keyed by `run_id` for the adapter (D6) to read. The
engine is NOT invoked by the integration adapter and NOT invoked by re-reading the
`serve_trial` return's `"manifests"` telemetry.

**Input:** the verified `ShardReadyManifest` set the sink accumulated for one
committed `run_id` (via `publish_shard`), triggered by `commit_trial`.

**Steps (in the engine, backend-independent):**
1. group shards by `(workflow_phase, direction, skip_mode)` using the manifest
   identity from D0 — **never** by parsing spool contents for identity;
2. for each of the four passes, read every verified shard's staged spool, verify
   byte count + SHA-256 (defense-in-depth even though Phase 4 verified), parse
   `s172_substripe_v1`;
3. build the four directional seed→match_rate maps:
   `forward_map_constant, reverse_map_constant, forward_map_variable,
   reverse_map_variable`;
4. **directional uniqueness invariant [TB-R1]** — see D2 (enforced here in the
   engine, exercised adversarially in D2's gate);
5. bidirectional populations = key intersections:
   `bidirectional_constant = fwd_const_keys & rev_const_keys`,
   `bidirectional_variable = fwd_var_keys & rev_var_keys`;
6. canonical enrichment per surviving seed using the **existing formulas**
   (source: the fuller constant/hybrid accumulator block in
   `window_optimizer_integration_final.py` ~640-682, **not** solely
   `_build_test_result_from_pw` which omits `bidirectional_selectivity`):
   ```python
   intersection_count        = len(fwd_keys & rev_keys)
   forward_only_count        = len(fwd_keys - rev_keys)
   reverse_only_count        = len(rev_keys - fwd_keys)
   intersection_ratio        = intersection_count / max(len(fwd_keys | rev_keys), 1)
   survivor_overlap_ratio    = intersection_count / max(len(fwd_keys), 1)
   intersection_weight       = intersection_count / max(len(fwd_keys) + len(rev_keys), 1)
   bidirectional_selectivity = len(fwd_keys) / max(len(rev_keys), 1)
   score                     = (forward_match_rate + reverse_match_rate) / 2.0
   forward_count             = len(fwd_map)   # trial-level count, per legacy
   reverse_count             = len(rev_map)
   bidirectional_count       = len(bidirectional_<mode>)
   ```
   window/offset/skip/trial/prng fields come from the D0 manifest metadata.
7. produce one canonical result object (shape may be optimized; carry at least):
   ```python
   @dataclass
   class MinerTrialAssembly:
       run_id: str
       bidirectional_constant: set[int]
       bidirectional_variable: set[int]
       forward_map_constant: dict[int, float]
       reverse_map_constant: dict[int, float]
       forward_map_variable: dict[int, float]
       reverse_map_variable: dict[int, float]
       canonical_records_constant: list[dict]
       canonical_records_variable: list[dict]
       binary_npz_path: str
       all_npz_path: str
       directional_counts: dict
       timing: dict
   ```

**Gate D1:** on a multi-shard, two-mode fixture with known populations, assert the
four maps, both intersection sets, and every derived field equal hand-computed
expected values. Must fail if any formula drifts or any pass is mis-grouped.

---

### D2 — Directional uniqueness invariant, fail-closed  **[TB-R1]**

Within one `(run_id, workflow_phase, accepted_attempt, family)` a seed appears
**at most once**. Stripes and sub-stripes partition disjoint seed ranges, so a
duplicate is a producer/coverage defect — **not** a dedup opportunity.

```python
if seed in this_directional_population:
    raise DirectionalDuplicateError(
        run_id, workflow_phase, direction, skip_mode, seed,
        first_stripe, first_sub_index, first_attempt, first_match_rate,
        dup_stripe,   dup_sub_index,   dup_attempt,   dup_match_rate)
```

The raise **aborts the trial before any canonical NPZ is written**. Phase 5 must
NOT resolve directional duplicates by max match_rate.

**Distinct from** the *global bidirectional/accumulator* dedup, which is the
legitimate highest-score-per-seed rule (`v1_4_4` §4.3, strict `>`, equal score
keeps prior) applied only at the cross-trial merge boundary in D3 — never inside a
directional population.

**Gate D2 [TB-R1 adversarial]:** craft two verified directional shards of the same
pass whose seed ranges overlap on one seed → assert Phase 5 raises
`DirectionalDuplicateError` with all identifying fields and writes **no** canonical
NPZ. A gate that dedups instead of raising is a failure.

---

### D3 — Parent-owned global merge / dedup / order / contract writer  **[§6.7.C, §12.1, §4.3]**

The **parent process is sole owner of global state.** Reuse the existing
*vectorized NumPy* accumulator model (`window_optimizer_integration_final.py`
~1749-1862): `searchsorted`-based highest-score-per-seed merge against the prior
`bidirectional_survivors_all.npz`, the rectangular-backfill guard
(`_dtype_for_field`), strict ascending seed sort, 22-array concatenation. Do NOT
replace it with concurrent Python dicts.

- global dedup: highest score per seed, equal score keeps prior (strict `>`);
- strict ascending seed order on the final arrays;
- 22-array contract wall (§12.1) runs on the **globally assembled** artifact only —
  a single shard cannot prove global uniqueness/order/equal-length;
- both canonical outputs — final `bidirectional_survivors_binary.npz` **and** the
  accumulator `bidirectional_survivors_all.npz` — pass the complete validator
  **before** the prior artifact is replaced;
- canonical finals use `np.savez_compressed` (frozen §12.1 writer behavior).

**22 arrays, exact order & dtype [`v1_4_4` §4.2] — freeze this list:**
```
seeds uint32 | forward_matches f32 | reverse_matches f32 | window_size i32 |
offset i32 | trial_number i32 | skip_min i32 | skip_max i32 | skip_range i32 |
forward_count f32 | reverse_count f32 | bidirectional_count f32 |
intersection_count f32 | intersection_ratio f32 | intersection_weight f32 |
bidirectional_selectivity f32 | forward_only_count f32 | reverse_only_count f32 |
survivor_overlap_ratio f32 | score f32 | skip_mode uint8 | prng_type uint8
```
`skip_mode`/`prng_type` via `utils/prng_encoding.py` (`encode_skip_mode`,
`encode_prng_type`) — hard-fail on unknown, no silent 0.

**Gate D3:** final + accumulator both pass the wall; wall FAILS on an injected
absent key / wrong dtype / unequal length / non-ascending seed / unknown-encoding
value. Assert highest-score-per-seed across a synthetic prior accumulator.

---

### D4 — `serial_reference` backend  **[§6.7.B]**

The correctness oracle, fallback, benchmark baseline, debug mode. Calls the shared
D1 engine + D3 merge directly, single process. One logical pass over records,
preallocated typed arrays, established defaults/encodings, **no** 22 independent
full-list comprehensions, no pandas.

**Gate D4:** on the D1 fixture, `serial_reference` produces arrays equal to the
hand-computed expected set. This is the oracle other backends are compared against.

---

### D5 — `process_sharded` backend  **[§6.7.A]**

Persistent, bounded, **CPU-only** process pool (`spawn` or `forkserver` — never a
post-thread/post-GPU fork). Each child receives ONLY a manifest
`{local_spool_path, expected_size, expected_sha256, stripe_id, sub_index,
trial_metadata}`, opens the staged spool itself, verifies bytes+sha, parses,
columnizes in one pass into typed partial arrays, shard-validates, writes an
**uncompressed** temp shard, returns ONLY a compact result manifest (paths, counts,
hashes).

**Prohibited (verbatim §6.7.A):** survivor dicts through `mp.Queue`; the 22 arrays
through pickle; a giant parsed JSON parent→child; 24 procs merely because Zeus has
24 threads. Children MUST NOT import torch/cupy or init a GPU context.

The **shared columnizer is identical** to D4's — both backends call the same
record→field conversion so they are provably equivalent. The parent still owns the
D3 global merge (children never touch global dedup state).

**Gate D5:** (a) probe asserts no survivor payload / no 22-array object crosses the
IPC boundary (only manifests do) and no child imports torch/cupy; (b) temp shards
are uncompressed; (c) `process_sharded` output arrays are `np.array_equal` to
`serial_reference` on the same input, per array.

---

### D6 — Thin `_build_test_result_from_miner` integration adapter  **[TB-R2]**

A **thin adapter**, not a second assembly. It reads the single `MinerTrialAssembly`
the **sink already computed and stored** for that `run_id` (produced in
`commit_trial`, per §1 + D1) and converts it to a `TestResult` + updates the
existing accumulator. It does **not** re-read the `serve_trial` `"manifests"`
telemetry, **not** reopen staged shards, and **not** re-run assembly. If it finds
itself building maps or reading spools, that is the double-assembly **[TB-R2]**
forbids — STOP.

Replace the incorrect
`window_optimizer_integration_final.py:415`
`_build_test_result_from_pw(miner_result, ...)` call with
`_build_test_result_from_miner(...)`. The old PWC builder's empty
`.get(..., default)` behavior must **not** remain as a silent fallback — if the
canonical assembly result is absent or incomplete, the miner path **fails
explicitly**, it does not return an empty `TestResult`.

**Gate D6 [TB-R2]:** (a) prove the *old* `_build_test_result_from_pw` returns zero
survivors for a nonempty staged miner run (documents why the current wiring is
broken); (b) the *new* adapter returns the expected populated `TestResult` with
counts matching `MinerTrialAssembly.directional_counts`; (c) absent/incomplete
assembly → explicit failure, never a silent empty result; (d) **no-double-assembly
probe** — with the sink's stored `MinerTrialAssembly` present, the adapter reads it
and performs **zero** shard-file opens and **zero** map construction (assert via a
spool-open counter / instrumented reader that the adapter path touches no staged
file). An adapter that re-reads shards fails this gate even if its output happens
to be correct.

---

### D7 — Contract-wall temp-cleanup + benchmark instrumentation  **[§12.1, §16.A, §17]**

1. temp shard files removed after successful final validation **and** coordinator
   acknowledgement (not before); assert none survive a completed trial and none are
   referenced after abort;
2. benchmark harness sweeping **1, 2, 4, 6, 8** assembly processes, reporting the
   eight stage timings (`gpu_execution_s, remote_spool_transfer_s,
   sha256_verification_s, json_parse_columnize_s, global_dedup_merge_s,
   final_npz_write_s, contract_validation_s, end_to_end_s`) plus peak RSS + swap
   watch. **Instrumentation only** — the §17 promotion decision and the ≥500K/≥25%
   gates are **Phase 6**, not here. Do not select a default backend in Phase 5;
   `serial_reference` stays default until Phase 6 measures.

**Gate D7:** cleanup assertions pass (no orphan temp/spool after commit; none
referenced after abort); the sweep runs and emits all eight timings + RSS for each
of {1,2,4,6,8} without OOM/swap-storm/unbounded-queue growth.

---

## 3. Global non-regression (every deliverable)

Phase 4 **63/63** and Phase 3 **17/17** stay green throughout. Any red there STOPS
work — report, do not "fix forward" into approved code without escalation.

## 4. Explicit stop conditions (STOP and report, do not code around)

- a required NPZ field cannot be produced from the manifest metadata + staged
  populations without inventing a formula → **STOP** (proposal amendment territory);
- a directional population proves incomplete/truncated/unassociable with its
  direction+mode at read time → **STOP** (producer/spool correction, not a Phase 5
  approximation);
- D0's manifest metadata cannot be made immutable+restart-durable within the
  seam-local change → **STOP** (do not expand Phase 4 scope unilaterally);
- any gate only passes by relaxing it to match a stub → **STOP**.

## 5. Deliverable → kickoff

Claude Code's kickoff is a one-liner pointing at this doc. Implement D0 first, iterate
its harness to green on 101, then STOP and report for review before D1.
