# CHAPTER_2_SOURCE_MAP_v1.md

**Chapter 2 (Bidirectional Sieve) — source map for a future reconstruction.**

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_SOURCE_GATHERING.md` (REV1).
**Type:** reconnaissance. **No** code, config or documentation was changed; nothing was
executed on a GPU; no pipeline, sieve or WATCHER was launched. Nothing here is Chapter 2 text.

**Box:** VM 101 (`zeus-ubuntu`, `192.168.3.177`), tree `/home/michael/distributed_prng_analysis`,
venv `~/venvs/torch`.
**HEAD at survey:** `73dbacf`. Working tree dirty — the concurrent Chapter-1 P0 session owns
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md` (modified), `scripts/extract_search_bounds_snapshot.py`
(untracked) and, per the brief, `window_optimizer.py` and `persistent_worker_coordinator.py`.
**Those four files were not opened this session.** Where a source depends on one, it is
recorded as a dependency, not as a finding.

**Date:** 2026-07-31.

---

## 0. Headline — read this before the table

**The full Chapter 2 is not missing. It is in git history, and the current file is the
result of a stale-copy overwrite.**

```
53a3829  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   709 lines   (chapters 1-12 suite, §1-13 + Summary)
d14dcdd  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   743 lines   (+ §14 inter-chunk cleanup)
         CHAPTER_2_BIDIRECTIONAL_SIEVE.md         34 lines   (root-level §14-only fragment, same commit)
248e48c  "chore: move CHAPTER docs to docs/ folder"
             D  CHAPTER_2_BIDIRECTIONAL_SIEVE.md          (-34)
             M  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md     (-709)   ← the chapter
         → docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md    34 lines
9ca2671 / c797045  + §15 (S146 PWC invariants), applied twice → 128 lines (current)
```

Verified this session by `git log --follow`, `git show <sha>:docs/…`, and
`git show 248e48c --name-status`. The "move" commit copied the **root 34-line fragment over
the 743-line docs/ chapter** and deleted the root file. This is the same defect class the
project has already named once — the stale-copy overwrite that silently reverted the
threshold fix at `2389b61` (skill §2.7 #2).

**Recovered section structure of the lost chapter** (`git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`):

| § | Title | Lines (in d14dcdd) |
|---|---|---|
| 1 | Mathematical Foundation (observable-data problem, collision space, sequential filtering, general probability formula) | 30-93 |
| 2 | Forward Sieve (what it does, algorithm, GPU implementation) | 94-163 |
| 3 | Reverse Sieve (**3.2 "Key Insight: Same PRNG, Different Direction"**) | 164-220 |
| 4 | Bidirectional Intersection (core principle, why powerful, what survivors mean) | 221-261 |
| 5 | Skip/Gap Handling (real-world problem, constant mode, variable/hybrid mode, survivor identity) | 262-325 |
| 6 | **Three-Lane CRT Architecture** (why mod 1000 / 8 / 125, lane disagreement = prune) | 326-376 |
| 7 | Architecture (component flow, data flow) | 377-422 |
| 8 | ROCm Environment Setup | 423-454 |
| 9 | `GPUSieve` Class (init, key methods, kernel caching) | 455-500 |
| 10 | Standard Sieve (`run_sieve`) | 501-544 |
| 11 | Hybrid Sieve (`run_hybrid_sieve`) | 545-587 |
| 12 | CLI Interface (job-file format, arguments) | 588-623 |
| 13 | Integration Points (pipeline position, inputs, outputs, consumed-by) | 624-659 |
| — | Summary + Version History | 660-712 |
| 14 | Inter-Chunk GPU Cleanup | 713-743 |

**Consequences for the reconstruction pass:**

1. It is a **restore-and-audit**, not a blank-page write. The historical §1-13 is a
   structural skeleton and, for §1/§3.2/§4/§6, likely still substantively correct — §6 in
   particular is the only known prose explanation of the triple-modulo test that is **live
   in the kernel today** (`prng_registry.py:984-986`, `:1042-1044`, `:3146-3148`) and is
   documented nowhere else in the current tree.
2. Every line of the recovered text is **pre-S172**: it describes `sieve_filter.py` /
   `GPUSieve` as the engine. RANGE-MINER — the certifying engine — did not exist. §7-§13 are
   therefore engine-specific and must be re-scoped, not restored.
3. The recovery is itself a finding worth reporting to Beta: "MISSING CORE CONTENT" is
   accurate about the file but the cause is a recoverable overwrite, which changes the cost
   and the risk profile of the repair.

---

## 1. Beta's nine derivation sources → authoritative live location

Completeness key: **COMPLETE** = a single authoritative live source exists and was read this
session. **PARTIAL** = located, but the authority is split, stale in part, or one leg is
unavailable. **NOT FOUND** = no authoritative source in the repo.

### Source 1 — Current forward and reverse residue construction — **COMPLETE**

| Route | Authoritative live source | What it says |
|---|---|---|
| **RANGE-MINER (certifying)** | `miner/range_miner_worker.py:538-580` `load_residue_window()`; called via `window_optimizer_integration_final.py:290-292` `_miner_residues_for_config()`; consumed at `:1214` | **One** derivation function, shared by parent and worker, session-filtered. Its docstring (`:557-560`) records the D6 defect it closes: the parent used to derive residues *without* the session filter while the worker applied it, so every single-session trial died on the `residue_sha256` check. Explicitly: "Do NOT reintroduce a second [derivation]." |
| Fingerprinting | `miner/range_miner_worker.py:523-527` `sha256_residues()` | sha256 over compact JSON of the int list — the contract for the coordinator-supplied `residue_sha256`. Content identity, not pathname. |
| Upload | `miner/range_miner_worker.py:812-814` | `cp.array(residues[::-1] if reverse else residues, dtype=cp.uint32)` |
| Legacy (`sieve_filter`) | `sieve_filter.py:230-235` (constant), `:393-398` (hybrid) | Same construct, per-process load. |
| PWC (`sieve_gpu_worker`) | `sieve_gpu_worker.py:188-191` | Same construct. |
| Dataset-side contract | `docs/DAILY3_CONSUMER_CONTRACT_v1.md` §4.1-§4.4, §7 | Index = position in the PRNG output stream; `offset` slices from index 0, i.e. the **oldest** end (§4.3); `load_residue_window` is named "the correctness-critical consumer" (§7). |

**A writer gets this from the miner worker, not from `sieve_filter.py`.** The legacy copies
are behaviourally identical but are not the certifying route.

### Source 2 — Host-side reverse ordering (`residues[::-1]`) — **COMPLETE**

**Confirmed still true in live source, on all three routes, this session:**

- `miner/range_miner_worker.py:813` — `residues[::-1] if reverse else residues`
- `sieve_filter.py:232`, `:395`
- `sieve_gpu_worker.py:189`
- Direction predicate: `miner/range_miner_worker.py:116-117` `is_reverse_family()` — a plain
  `family_name.endswith("_reverse")`.

**Confirmed that reverse kernels iterate the PRNG forward:** `java_lcg_flexible_sieve`
(`prng_registry.py:958-1004`) and `java_lcg_reverse_sieve` (`:3115-3169`) are the same
recurrence `state = (a*state + c) & m` step for step (forward `:975/:978/:982/:988`, reverse
`:3135/:3139/:3143/:3153`). No modular inverse, no backward recurrence anywhere. The reverse
kernel differs only in hardcoding `a`/`c` in the body (`:3125-3126`) instead of taking them as
arguments — which is why the reverse-constant ABI is 12 args with no family tail
(`miner/range_miner_worker.py:207-211`, `:224-226`).

Skill §0.2 is correct and the code matches it. **This is the single most misread fact in the
subject area** and belongs early in the chapter.

**Documentation hazard the reconstruction must resolve, not inherit:** two registry
descriptions state the implemented behaviour ("Fixed skip **forward** validation",
`prng_registry.py:4099`, `:4118`) while the java_lcg ones say "fixed skip **backward**
validation" (`:3911`, `:3917`) for kernels whose bodies are forward recurrences. The
whitepaper's §4 predicate `G(s,−i)` (`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:58`)
also reads as a backward step. See §3 of this map.

### Source 3 — Constant vs hybrid kernel semantics — **COMPLETE**

**Cite, do not re-derive:** `docs/HYBRID_SKIP_BOUND_AUDIT.md` (376 lines, commit `808e19b`) —
§2.4 traces the RANGE-MINER (certifying) consumer chain, §3.1 answers whether
`expected_skip = 5` is a constant/default/derived, §3.2 the signature difference, §4 whether
the hybrid algorithm is compatible with a skip range at all, §7 the recommendation
(**wire in, not remove**).

Verdict as stated there and confirmed live this session: **22/22 constant kernels declare
`int skip_min, int skip_max`; 0/22 hybrid kernels do.**

Live anchors a writer needs:

| Fact | Anchor | Reading |
|---|---|---|
| Constant kernel takes skip bounds and searches them | `prng_registry.py:963` (signature), `:972` (`for skip = skip_min..skip_max`), `:992-995` (keep best rate + best skip) | One fixed stride per seed, best over the configured range. |
| Constant skip semantics | `prng_registry.py:974-990` | `offset` pre-advance, then a `skip` burn, then per draw: advance 1 → extract → advance `skip`. |
| Hybrid forward kernel signature has **no** `skip_min`/`skip_max` and **no** `offset` | `prng_registry.py:1007-1012` (15 params, ends `float threshold, unsigned long long a, unsigned long long c`) | The trial's skip range cannot constrain the hybrid pass. |
| Hybrid forward hardcodes the stride estimate | `prng_registry.py:1027` `int expected_skip = 5;` | Adapted after each hit (`:1048`); search window `[expected_skip − tol, expected_skip + tol]` (`:1033-1034`). |
| What `strategy_tolerances` actually does | `prng_registry.py:1023` (`skip_tolerance = strategy_tolerances[strat_id]`) → `:1033-1035` | It is the **half-width of the per-draw skip search around the running estimate**. It is the only skip-range concept the hybrid kernel has. |
| What `strategy_max_misses` does | `prng_registry.py:1022` → `:1055-1058` | Consecutive-miss abort for that strategy. |
| Hybrid forward scans all strategies, keeps the best | `prng_registry.py:1061-1067` | vs hybrid **reverse**, which returns on the first strategy clearing threshold (`:3229-3240`). |
| `skip_sequences` is the per-draw stride record | `prng_registry.py:1054` (`current_skip_seq[draw_idx] = actual_skip`), `:1075-1077` (emit `skip_sequences[pos*k + i]`) | Buffer allocated `n_seeds × k` uint32 at `miner/range_miner_worker.py:843`. |
| Where the sampled bounds die on the certifying route | `miner/range_miner_worker.py:776` (unpack `skip_range`) → `:871` (into `BuildContext.skip_min/skip_max`) → `_hybrid_prefix()` `:177-193` **never emits them** | Values survive argparse, config, coordinator, ledger, manifest, payload, worker unpack and the arg-build context, then die one call before launch. |
| Constant path *does* emit them | `miner/range_miner_worker.py:171-172` inside `_constant_prefix()` | The asymmetry is in the arg builder, not the payload. |

**Why skip exists at all** — the physical model (pre-test draws, per-session equipment
selection, interleaved games) is skill §0.4 and is **not written down in any repo document**.
See §6 Gap G-1.

### Source 4 — Threshold and skip propagation — **COMPLETE (by citation)**

**Cite, do not re-derive.**

- Thresholds: `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (384 lines) — §2 Route A
  hop-by-hop, §4 Route B, §5 the miner's post-D6 path as the contrast case, §9 "where exactly
  the value is dropped", §10 regression forensics, §11 VIR declaration.
- Skip: `docs/HYBRID_SKIP_BOUND_AUDIT.md` (see Source 3).
- Repair record: `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` (419 lines, commit
  `8a55a68`).

**Post-`8a55a68` live state, confirmed this session:** one canonical resolver
`resolve_directional_threshold()` at `window_optimizer_integration_final.py:210`, used by both
routes at `:2363-2364`. Miner side: the parent resolves direction per stripe via the §6.8
phase table (`miner/range_miner_coordinator.py:1491-1523`, `:3410-3430`) and stamps
`min_match_threshold` into the assignment payload; the worker does **not** choose a threshold
and does **not** know about forward/reverse (`miner/range_miner_worker.py:777-784`). The
`0.25` legacy default survives only for pre-D6 payloads and is caught by the D6 gate, not by
silent fallback. Contradictory `min_match_threshold` / `phase2_threshold` pairs fail closed
(`miner/range_miner_worker.py:785-798`).

**Provenance triple** — requested / payload / effective:
`miner/range_miner_coordinator.py:1644-1646` (the definition), `:3134-3153` (recording
per-substripe and per-stripe effective values), `:3177-3224` (assembly), `:3234-3307` (the
fail-closed enforcement: effective MUST be present; all sub-stripes of a stripe MUST agree;
disagreement or absence is a certification failure). `:1335-1343` states the principle — the
physical evidence that the requested threshold reached execution.

**PARTIAL leg (dependency, not a gap):** the PWC hybrid quarantine
(`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`) lives in `persistent_worker_coordinator.py`,
which the concurrent P0 session may be editing. It was **not opened this session**. Its
behaviour is pinned by `tests/test_s172_threshold_propagation.py:662-689` and described in the
repair report; PWC is non-certifying (skill §0.7, §3), so the chapter should describe it as a
quarantined diagnostic path and cite, not trace, it.

### Source 5 — Session-stream rules — **PARTIAL**

What exists and is authoritative:

- `docs/DAILY3_CONSUMER_CONTRACT_v1.md` (514 lines) — §4 "Ordering and indexing — the
  highest-risk surface" (§4.1 what the file actually is, §4.2 index = PRNG stream position,
  §4.3 offset slices from the oldest end, §4.4 **reverse-sieve direction is derived from array
  order**, §4.5 two contradictory intra-day orderings coexist, §4.6 positional-tail consumers);
  §5 session handling and the split files (§5.1 how a session is distinguished, §5.4 the S119
  claim CONFIRMED and unenforced); §8 implicit invariants nothing validates; §10 defects
  observed.
  Named defect a writer must not restate as safe: `:418` — the intra-day sort key is
  `(date, session)` raw (**evening first**), and switching to true chronological order changes
  every combined-session residue window and every `residue_sha256` with no error anywhere.
- `docs/TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` — `:34`, `:68`, `:77-80`,
  `:188-195`. This is Alpha's **submission**, i.e. the argument, not the ruling.
- The **binding ruling** — per-session ordering normative, combined-container order carrying
  no PRNG-advance meaning, combined-session sequential sieve non-certifying and prohibited by
  default, production re-optimization per-session, chronological-reorder migration cancelled —
  is carried in-repo only as the summary in `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10 (tracked
  at `2eb91ab`).

**Why PARTIAL:** there is no `TB_RULING_*` document for the 2026-07-30/31 dataset-lifecycle
rulings, though there is one for every other adjudicated area
(`TB_BINDING_RULINGS_S172_PHASE4.md`, `TB_RULING_S176/S177/S178/S179_*`). The chapter would be
citing a skill summary as the authority for a binding constraint. See §6 Gap G-2.

### Source 6 — Seed-domain partitioning — **COMPLETE**

| Route | Source | What it says |
|---|---|---|
| **RANGE-MINER, macro** | `miner/range_miner_coordinator.py:245-268` `partition_macro_stripes()` | Contiguous macro-stripes over `[base_start, base_start + total_seeds)`, **no gap, no overlap**. A macro-stripe MAY exceed one GPU cap. Default `miner_stripe_size = 67_108_864` (`:127`). |
| **RANGE-MINER, sub** | `miner/range_miner_worker.py:472-486` `select_seed_cap()`, `:493-503` `partition_stripe()` | The **worker** partitions its one assigned macro-stripe into GPU-safe sub-stripes at runtime. Cap branches on backend (`rocm` → AMD caps, `cuda` → NVIDIA caps). |
| Cap agreement | `miner/range_miner_coordinator.py:224-243` `advertised_effective_cap()`, `:270-277` `expected_substripes_for()` | The coordinator sizes `expected_substripes` with the **same** cap the worker will partition with (Blocker 7: macro sizing ≠ sub-stripe sizing). |
| Coverage proof | `miner/range_miner_coordinator.py:311-322` `_coverage_exact()`, `:325-381` `evaluate_stripe_completion()` | L8: a stripe is complete only when `substripes_done == expected == distinct(sub_index)`, seed-count sums match, survivor sums match, **and** sub-stripe ranges tile the parent exactly (gap or overlap → not complete). |
| Ledger cardinality | `miner/range_miner_coordinator.py:20-23`, `:85-90`, `:416-431` | SHARD-level, keyed `(run_id, stripe_id, attempt, sub_index)`; never one-row-per-stripe. |
| Legacy (contrast) | `coordinator.py:1415-1447` | `base_chunk_size = min(total_seeds // num_workers, cap)`, chunks walked from seed 0; job spec carries `skip_range: [args.skip_min, args.skip_max]`, `search_type: 'residue_sieve'`. Note it hardcodes `'sessions': ['midday','evening']` at `:1444` — a combined-session assumption on a non-certifying path. |

### Source 7 — RANGE-MINER execution — **COMPLETE**

Modules and sizes read/enumerated this session:

```
miner/range_miner_coordinator.py   4304   stripe ledger, state machine, macro partitioning, L8 reconciliation,
                                          §6.8 phase table, threshold payload + provenance enforcement
miner/range_miner_worker.py        1449   READY handshake, sub-stripe loop, per-family kernel ABI builders,
                                          residue-window authority, inline-vs-spool transport
miner/range_miner_npz_writer.py    1290   Phase-5 assembly: spool validation, canonical replay loop, trial assembly
miner/assembly_shard_worker.py      887   process_sharded shard body
miner/assembly_backends.py          393   frozen two-backend interface (serial_reference | process_sharded)
miner/step1_ingress.py              288   Step-1 accumulator ingress + certified-path resolution
miner/range_miner_protocol.py       265   8 message types, length-prefixed JSON framing
```

Lifecycle anchors: assignment attempt pairing `:192-208`; deferred queue (capacity/slot
waiters) `:147`; retry matrix implemented per workflow phase 1-4 `:2866`; four-stage
`test_both_modes` family/phase driver `:4094`, `:4260`. Worker side: cap selection `:1222`,
sub-stripe loop `:1234-1254`, per-sub result build `:1284-1306`, inline-vs-spool by size
`:1324-1327` (`INLINE_BYTE_LIMIT = 48 MiB`, `:956`), atomic spool write `:994-1001`.

The kernel-ABI table is the part a chapter most needs and is the least documented elsewhere:
`miner/range_miner_worker.py:160-174` (`_constant_prefix`, 11 elements), `:177-193`
(`_hybrid_prefix`, 13 elements), `:196-202` (offset tails), `:206-212` (the comment recording
that forward and reverse **constant** kernels do not share arg layout), then per-family
builders from `:214`. Arity per variant is stated inline: reverse-constant 12, reverse-hybrid
14, java_lcg forward-hybrid 15, lcg32 forward-hybrid 17.

### Source 8 — D5/D6 assembly and provenance contracts — **COMPLETE**

| Contract | Source | What it says |
|---|---|---|
| Backend interface, frozen | `miner/assembly_backends.py:12-15`, `:118-121`, `:344-360` `get_assembly_backend()` | Two backends declared in full, closed set, **no silent default** — resolution fails closed; `process_sharded` "must NOT be added later" (it was declared up front). |
| `serial_reference` roles | `miner/assembly_backends.py:45-51` | Four documented roles including FALLBACK; `process_sharded` becomes production default only on a ≥20% median improvement. |
| `process_sharded` status | `miner/assembly_backends.py:273-284` | Implemented, **no default precedence**, `serial_reference` remains production default. Matches skill §2.8 ("Available, UNPROMOTED"). |
| Measurement parity | `miner/assembly_backends.py:131-198`, `:292-297` | Both backends measured identically; the metadata gauntlet runs in the same precedence. |
| Phase-5 assembly | `miner/range_miner_npz_writer.py:516-545` projection build, `:700-716` `read_and_validate_spool`, `:847-863` **the canonical replay loop**, `:863` `merge_validated_spools`, `:975` `prepare_trial_assembly`, `:1080` `assemble_trial`, `:1148-1268` sink + `commit_trial`/`abort_trial` | Spool-local validation only in shards; parent alone owns merge/dedup/intersection. |
| Canonical error replay | `miner/range_miner_npz_writer.py:614-680` | `CANONICAL_SPOOL_READ_ERRORS` is a closed map; an uncaptured error class may not be replayed. |
| 22-array contract | `utils/canonical_arrays.py` `CANONICAL_ARRAY_CONTRACT` (:98-123); record side `utils/canonical_records.py` `CANONICAL_RECORD_FIELDS`, re-exported deliberately at `miner/range_miner_npz_writer.py:64`, `:177` | Frozen names/order/dtypes. Only 4 columns carry per-seed information (skill §2.3). |
| Ownership separation | `tests/test_s172_phase4_coordinator.py:1581-1600` gate 21 | The coordinator module owns **no** Phase-5 assembly: forbidden tokens include `range_miner_npz_writer`, `EXPECTED_NPZ_KEYS`, `np.savez`, `np.load`, `import numpy`, `process_sharded`. |
| Step-1 ingress | `miner/step1_ingress.py:143` `resolve_assembly_backend`, `:157` `build_assembling_sink`, `:168` `require_assembly`, `:205` `ingest_assembly`, `:265` `certified_paths` | Miner candidates reach the Step-1 accumulator and a certified generation. |
| Finalizer (L2/L3) | `utils/run_finalizer.py:690` `_l2_sort_key`, `:714` `_select_l2_winners`, `:752-808` L3 merge, `:811-827` global seed-ascending sort, `:130` `BINARY_NPZ_NAME` | **FROZEN — import, never fork** (skill §4). Same-trial/same-mode collision raises `AccumulatorConsistencyError`. |
| Threshold provenance | see Source 4 (`miner/range_miner_coordinator.py:3177-3307`) | requested / payload / effective, fail-closed. |

`serial_reference` vs `process_sharded` is therefore documented **in code comments of
`miner/assembly_backends.py`**, which is the authoritative source; there is no separate
contract document. `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md` and `…_D5.md` are the
briefs that specified them and are useful as intent, not as as-built.

**One naming trap for the writer:** `EXPECTED_NPZ_KEYS` (CLAUDE.md §6 Phase 5, "NPZ contract
wall §12.1") exists in the tree **only** as a forbidden-token string in
`tests/test_s172_phase4_coordinator.py:1589`. There is no symbol by that name. The contract
wall is implemented under the `utils/canonical_arrays.py` / `utils/canonical_records.py`
names. Do not write the chapter around a symbol that does not exist.

### Source 9 — Independent bounded known-answer controls (Beta's Wall C) — **NOT FOUND (raw materials only)**

**No Wall C specification, harness, or fixture exists in the repo.** Every in-repo mention is
a deferral:

- `docs/TFM_PROJECT_FACTS_SKILL.md:325` — "(C) **bounded independent known-answer
  correctness** — a reference that does NOT call the miner's coordinator/backend/finalizer".
  This one line is the entire specification.
- `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md:144` — "Do not build Phase 6's
  known-answer fixtures here — that is Beta's Wall C, its own item."
- `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md:366` — same deferral.
- `docs/PHASE6_PREREQS.md` (REV3) — no Wall C section; it covers the D3.5 publication
  filesystem, code/env parity, PWC/ZMQ reachability and the four infrastructure items.

**What exists today that could serve as raw material** (all read this session):

| Candidate | Location | Fit for Wall C |
|---|---|---|
| `create_java_lcg_test.py` | root, tracked | **Best fit for constant skip.** Known seed 1234, skip 5, 512 draws; loop is `advance skip → advance 1 → emit`, which is **sequence-identical** to the kernel's `offset` pre-advance + skip burn + per-draw (advance 1, emit, advance skip) at `offset=0`. Writes `test_multi_prng_java_lcg.json` — **the file is not present in the tree**; it must be regenerated. |
| `create_java_lcg_variable_test.py` | root, tracked | Known seed 1234, cyclic skip `[5,5,3,7,5,5,8,4]`, 512 draws — i.e. exactly the variable-skip pattern skill §0.4 uses as its example. Fit for the hybrid path, **once hybrid skip bounds are wired** (skill §2.7 #4, §8: hybrid certification is blocked until then). |
| `prng_registry.py:170-183` `java_lcg_cpu` + `'cpu_reference'` registry entries (`:3960`, `:3973`, and 18 others) | tracked | **Not a drop-in oracle.** It applies `skip` **once** before the loop then emits `n` consecutive outputs; the kernel applies `skip` **between every draw** (`:987-989`). The two agree only at `skip = 0`. A Wall C built on this without correcting the skip model would be a check that is not checking. |
| `synthetic_draw_injector.py:205-240`; `generate_fingerprint_library.py:427-454` | tracked | Consume `get_cpu_reference()` and inherit the same skip-model mismatch. |
| `tests/smoke_s172_phase5_d6_zeus_single_gpu.py` | tracked, 1094 lines | A **platform-parity** harness (Wall B), not a known-answer one: it compares two runs of the same code on different silicon. Its own header (`:96-97`, `:174-188`) is explicit that the coordinator, assembly, finalizer and writer are common-mode on both legs — which is exactly what Wall C must exclude. |
| `tests/test_s172_threshold_propagation.py:98-115` | tracked | The correct **pattern** to copy: hand-transcribed literal ORACLES that are "never imported from a module under test". |

**This is the most valuable single output of this pass:** Wall C must be written from live
code and first principles. Nothing in the repo can be assembled into it, and the one
convenient-looking artifact (`cpu_reference`) is a trap.

---

## 2. `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` — assessment

**Verdict: PARTIALLY USABLE — high value, bounded scope, one materially superseded section.
Use it; do not paste it.**

**What it is.** 1205 lines, untracked, dated 2026-07-28, surveyed at `42a7229` (HEAD is now
`73dbacf`). Read-only descriptive trace with `file:line` on every claim and `[inferred]`
markers on inference. Sections: §1 naming, §2 control flow, §3 what "reverse" computes, §4
intersection, §5 the 22 NPZ fields provenance-by-field, §6 `forward_matches`/`reverse_matches`
at the source, §7 kernel coverage reality, §8 thresholds, §9 selection and merge, §10 31
numbered observations, §11 artifacts, §12 files-read declaration.

**Why it is usable.** I checked its staleness by diff rather than by reading:
`git diff --stat 42a7229..HEAD` shows the only **non-test, non-doc** sources changed since the
survey are `miner/*`, `window_optimizer_integration_final.py` (+895) and
`persistent_worker_coordinator.py` (+83). **`sieve_filter.py`, `sieve_gpu_worker.py`,
`prng_registry.py`, `coordinator.py`, `convert_survivors_to_binary.py` and `utils/*` are
byte-unchanged since the trace.** Every anchor the trace places in those files is still
valid — I spot-verified `prng_registry.py:958-1004`, `:1005-1081`, `:1027` and the
`residues[::-1]` sites directly.

**What it gives the reconstruction that nothing else does:**

- §5 — the 22 NPZ columns traced field by field with an origin classification (PER-SEED /
  TRIAL-AGG / CONFIG / CATEGORICAL) and a tally: **4 of 22 carry per-seed information**. This
  is the skill's §2.3 claim, derived rather than asserted.
- §6.5 — the value space of `forward_matches` is `matches/k`, so at most `k+1` distinct
  values; with `window_size ≤ 50` the converter's low-variance warning
  (`convert_survivors_to_binary.py:194-195`) fires for any population above ~510 **regardless
  of correctness**. A chapter that documents that column without this is misleading.
- §3.1/§3.2 — the forward-vs-reverse kernel comparison table, and the plain-language framing:
  "a *time-reversed target*, not a time-reversed generator."
- §7.4 — exactly **4 of the 44** registry entries are ever compiled in production, while the
  registry's size of 44 is load-bearing for the uint8 `prng_type` encoding
  (`utils/prng_encoding.py:42-43`, pinned by `tests/test_prng_encoding.py`).
- §10 O1-O31 — a defect/oddity inventory the chapter can decide to mention or omit
  deliberately rather than by omission.

**What is superseded — do not carry forward:**

1. **§8.5 and O7 ("Optuna's suggested thresholds never reach a kernel") are FIXED.** The
   trace was written at `42a7229`, before `8a55a68`. Both routes now resolve through
   `resolve_directional_threshold()` (`window_optimizer_integration_final.py:210`, used at
   `:2363-2364`). Anything the trace says about `ft`/`rt` defaulting is history.
2. **§8.4 and O9 ("variable-skip runs at a hardcoded 0.50")** — still true of the legacy and
   PWC paths, but the PWC hybrid path is now **quarantined**
   (`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`, skill §2.7 #3) and PWC is non-certifying.
   The statement must be re-scoped, not repeated.
3. **§2.7, §5, §9, §11** cite `window_optimizer_integration_final.py` line numbers that moved
   under +895 lines of D6 work. Treat every anchor in that file as needing re-verification.
4. **The miner is absent by design.** The trace declares it out of scope (`:10-15`, `:1203-1206`).
   Sources 6, 7, 8 and half of 1 and 4 have **no coverage** in it. Since the miner is the
   certifying engine, the trace covers the *non-certifying* half of the chapter's subject.

**Recommended disposition:** the trace is a legitimate prior-art input and should be **tracked
in git** so the reconstruction can cite it stably (it is currently untracked and would be lost
by any clean checkout). It should not be promoted to Chapter 2 and should not be edited into
Chapter 2 wholesale — its scope is the complement of what certifies.

---

## 3. Whitepaper-vs-chapter boundary

`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` is 167 lines, §1-§10, pure mathematics,
no `file:line`, no engine. Proposed division:

| Belongs to the **whitepaper** (cite, never restate) | Belongs to **Chapter 2** (implementation) |
|---|---|
| §3 forward predicate `F(s) ≥ τ_f`; §4 reverse predicate; §5 `P(B) ≈ P(F)²`; §6 exact-match limit `10⁻³⁰⁰` at n=50 | How `F` and `R` are actually computed: the kernel loop, the triple-modulo lane test, `matches/k`, the best-over-skip maximisation |
| §7 why loose thresholds are required (no variance → no ranking → no learning signal) | Where a threshold physically enters the kernel, who resolves it, and how "effective" is proven (Source 4) |
| §8 ML after sieving is statistically sound; §9 autonomy adjusts parameters, never structure | Which parameters are tunable, which reach silicon, and how a dead dimension is detected (skill §0.5) |
| The exponential-collapse *argument* | The set-intersection *mechanism*: `forward_set & reverse_set` — no joint gate, no re-verification of the pair, no combined-rate threshold |

**The boundary line, stated once:** *the whitepaper says why bidirectional sieving works; the
chapter says what this system does when it runs one.*

**The gap the boundary must not leave — and the one place the two genuinely disagree.**

The whitepaper's §4 predicate is `R(s) = (1/n) Σ 1[G(s,−i) = d_{n+1−i}] ≥ τ_r`
(`:57-59`) — a generator evaluated at a **negative index**. The implementation evaluates
`G(s, i)` (positive index, forward recurrence, `prng_registry.py:3143`) against a **reversed
residue array** (`sieve_filter.py:232`, `miner/range_miner_worker.py:813`). The draw term
matches; the generator term does not. Consequently, for one seed the forward and reverse
passes generate the **identical** output sequence and differ only in what they are compared
to — whereas §4's independence premise (`:61-62`), which §5's squaring (`:79`) rests on, is
stated about a construction where they would not.

The chapter must **state the implemented construction plainly and point at the whitepaper
section that assumes a different one.** It must not silently adopt the whitepaper's notation
(that reintroduces the "reverse means backward" misreading skill §0.2 exists to prevent), and
it must not assert the statistical consequence either way — that is a mathematics question,
and mathematics is the whitepaper's side of the boundary. The correct chapter sentence is
descriptive plus a forward reference; resolving it is a separate item for Beta.

---

## 4. §14 disposition — **RETAIN, corrected and re-scoped**

What §14 documents (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:4-36`): inter-chunk
`_best_effort_gpu_cleanup()` in the two forward-sieve loops of `sieve_filter.py`, added
2026-01-26 to stop VRAM-fragmentation GPU hangs across ~26 chunks per Step-1 invocation;
20/20 benchmark trials clean, <5% overhead.

**Retain, because:**

- The content is still accurate. The guard text `if chunk_start + chunk_size < seed_end:`
  still matches live source; only the line numbers moved (§14 says "lines 230, 385"; the calls
  are now at `sieve_filter.py:326-327` and `:481-482`). `sieve_filter.py` is unchanged since
  the descriptive trace verified this.
- It is a real historical fault-mode record on a path that still runs (legacy/local chunks).
- Deleting a documented fix invites its removal from code (skill §0.4 standing rule).

**Corrections required when it is folded back in:**

1. Update the line references, or drop them in favour of the function name.
2. Re-scope: §14 is about the **legacy chunked** engine. RANGE-MINER's persistent per-GPU
   daemons exist precisely because launch-storm behaviour caused `GCVM_L2_PROTECTION_FAULT` on
   the rigs (skill §0.7), and Phase 6.0 recorded **no GPU reset and no `GCVM_L2` fault** on
   both platforms (skill §2.8). §14 must not read as a current mitigation for the certifying
   engine.
3. It belongs at the end, after the restored §1-13 — not as the chapter.

**§15 (S146 Persistent Worker Execution Path) — SUPERSEDE.** It is present **twice verbatim**
(`:38-83` and `:85-128`), a duplication also flagged in `docs/CHAPTER_1_AUDIT_v1.md` §"Appendix
B / C" for the sibling chapter. Its subject is PWC, which Beta retired from certifying
authority on 2026-07-31 (skill §0.7, §3), with the hybrid path additionally quarantined. Both
copies should be replaced by a single short "non-certifying diagnostic paths" subsection that
cites the retirement.

---

## 5. Citation inventory — cite, do not duplicate

| Document | Lines | Cite it for |
|---|---|---|
| `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` | 167 | All of the mathematics (§3-§9). See §3 boundary above. |
| `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` | 384 | The whole threshold path, Route A/B, the drop point, regression forensics. **Do not re-derive.** |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` | 376 | 22/22 vs 0/22, `expected_skip = 5`, whether a skip range is even compatible with the hybrid algorithm, wire-in-not-removal. **Do not re-derive.** |
| `docs/DAILY3_CONSUMER_CONTRACT_v1.md` | 514 | Dataset schema, ordering, `load_residue_window` requirements, session split, the unenforced S119 claim. **Do not re-derive.** |
| `docs/CHAPTER_1_AUDIT_v1.md` | 732 | The audit method to imitate; §4 dead-dimension inventory; C-1 (skip bounds documented for variable, rejected by hybrid kernels); C-2 (**`offset` has three incompatible definitions** — Chapter 2 must pick one and say which); C-5, C-6. **Do not re-derive.** |
| `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` | 292 | Binding pipeline map; §7 the data-contract spine and RANGE-MINER's closed question; §7.3 three pre-existing defects not caused by the miner. **Binding — do not restate differently.** |
| `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md` | 419 | What `8a55a68` actually repaired. |
| `docs/S172_SIEVE_PATH_VERIFICATION_SCOPE.md` | 100 | The four sieve paths × 6 covered families = 24 variants; "two DIFFERENT claims — do not conflate"; per-path verification status. Directly reusable framing for a chapter section on what is and isn't proven. |
| `docs/D6_RELEASE_GRADE_CERTIFICATION_RECORD.md`, `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md` | — | The authoritative generation (`gen-20260730T002104136270Z-step1_java_lcg_0`, `b08c2c5`, `artifact_sha256 0e0092fe…c4b0`) and the CUDA/ROCm byte-identity result. |
| `docs/VERIFICATION_INTEGRITY_STANDARD.md` | 159 | VIR-1…6, if the chapter makes any verification claim. |
| `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` | 1205 | The legacy/PWC engine, the 22-column provenance table, kernel coverage reality. **Untracked — track it before citing.** |
| `git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | 743 | The lost chapter itself. §6 (three-lane CRT) especially. |

---

## 6. Gaps — Beta's nine sources with no authoritative source in the repo

*This is the section that tells the reconstruction what must be written from live code.*

**G-1 — Why skip exists. No document. (affects Sources 3, 6)**
The physical model — two unpublished pre-test draws before every live draw, per-session
equipment selection by an auditor-verified RNG program, and evening co-drawing of D3/D4/
Fantasy 5/Daily Derby so other games' outputs sit between observable Daily 3 values — exists
in the project only as skill §0.4, which states outright that it is "the part nobody had
written down." The primary source is the *California State Lottery Daily & SuperLotto Plus
Draw Procedures* (eff. 2021-06-09), which is **not in the repo**. This absence already caused
Alpha, Beta and Claude Code to independently recommend deleting `skip_min`/`skip_max`.
**Chapter 2 §5 (Skip/Gap Handling) is the natural home for it.** Writing it there is arguably
the highest-value paragraph in the whole reconstruction.

**G-2 — The binding session-stream ruling has no ruling document. (Source 5)**
Per-session ordering normative / combined-container order carrying no PRNG-advance meaning /
combined-session sequential sieve non-certifying / per-session re-optimization / reorder
migration cancelled — carried in-repo only by `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10. Alpha's
`TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` is the request, not the ruling.
Every other adjudicated area has a `TB_RULING_*` file. The chapter should cite a ruling
document; one needs to exist.

**G-3 — Wall C has a one-line specification and no artifact. (Source 9)**
See Source 9 above. Must be written from live code. The `cpu_reference` functions are a trap:
`java_lcg_cpu` (`prng_registry.py:170-183`) applies `skip` once before generating; the kernel
applies it between every draw (`:987-989`). They agree only at `skip = 0`. The usable seeds
are `create_java_lcg_test.py` (constant, sequence-correct) and
`create_java_lcg_variable_test.py` (variable), neither of which is a harness and neither of
whose output JSON files is present in the tree.

**G-4 — No document describes the miner's kernel-ABI-by-variant contract. (Sources 3, 7)**
The per-family arg builders (`miner/range_miner_worker.py:160-350+`) encode a real, non-obvious
contract — forward and reverse constant kernels do **not** share arg layout; every fixed-skip
reverse kernel hardcodes its generator parameters in the body; arity varies 12/14/15/17 by
variant and family. This is currently documented only in the code's own comments. It is
directly load-bearing for anyone reasoning about the sieve and belongs in Chapter 2.

**G-5 — The three-lane CRT test is live in code and documented nowhere current. (Source 1/3)**
`(output % 1000) && (output % 8) && (output % 125)` appears in every kernel
(`prng_registry.py:984-986`, `:1042-1044`, `:3146-3148`). The only prose explanation is
§6 of the **deleted** chapter (`d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:326-376`).
Recover it, then re-verify it against the kernels — it was written pre-S172.

**G-6 — `serial_reference` vs `process_sharded` has no as-built contract document. (Source 8)**
Authority is `miner/assembly_backends.py`'s module docstring and class comments. The D4/D5
briefs record intent, not as-built. Not a blocker — the code is unusually well-commented — but
the chapter is where the as-built statement should land.

**G-7 — Chapter 2's own intended scope is recorded in a file this session may not read.**
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1315` — "**Chapter 2: Sieve Filter (Step 2)** will
cover: …" — is owned by the concurrent P0 session and was **not opened**. The reconstruction
pass should read it once that session lands. Noted as a dependency, not a gap in the repo.

---

## 7. Stale / duplicate sieve documentation and source — named, not deleted

**Do not delete any of these.** Per the standing ruling of 2026-07-31 on stale duplicate
source files, the known duplicates are left alone deliberately. This list exists so a future
reader does not mistake one for current.

### Documentation

| File | Status | Why a reader could be misled |
|---|---|---|
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | **Fragment (current file)** | 128 lines: §14 once, §15 twice verbatim. Already ruled MISSING CORE CONTENT. Skill §3 lists it as superseded. |
| `docs/chapter2_interchunk_cleanup_section.md` | **Duplicate source of §14** | 46 lines, titled "Section to add to CHAPTER_2_BIDIRECTIONAL_SIEVE.md". Contains §14 in `###` heading depth plus one extra sentence the chapter copy dropped (`sieve_filter.py` only called cleanup at script exit, "line 679"). Superseded by the in-chapter copy. |
| `agent_contexts/step2_bidirectional_sieve.md` | **Not a chapter — an LLM prompt template** | 119 lines: "MISSION STATEMENT", "DECISION RULES", "DECISION FORMAT". Its §"Skip Hypothesis Testing" and §"Sequential Filtering Power" read like chapter content but this is agent-context scaffolding, not documentation of as-built behaviour. |
| `docs/DOCUMENTATION_AUDIT_20260131.md:93-99` | **Stale audit entry** | Rates Chapter 2 as a "LOW / single line" fix (add `rig-6600c` to the ROCm prelude) — assessed against the 743-line version, before anyone noticed §1-13 were gone. |
| `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` | **Untracked; partially superseded** | §8.5/O7 fixed by `8a55a68`; miner out of scope. See §2. Risk is specifically that it is *good*, so a reader may over-trust its threshold sections. |
| `docs/PROPOSAL_NPZ_Auto_Conversion_Step2.md`, `docs/PROPOSAL_STEP2_OBJECTIVE_FUNCTION_v1_3_0.md`, `docs/TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`, `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` | Proposals / ruling requests | Intent documents, not as-built. Not re-read this session — listed so the reconstruction checks their disposition rather than citing them as current. |

### Source files that look like the sieve and are not the live path

| File | Status |
|---|---|
| `sieve_filter_INTEGRATED.py` | Duplicate of `sieve_filter.py` lineage; contains its own `residues[::-1]` at `:136`, `:306`. Not on any live path. |
| `reverse_sieve_filter.py` | A second reverse engine the bidirectional path never invokes. Its own header warns it "should NOT be run directly"; it does **not** reverse the residue array (`:160`); its 11-argument kernel call does not match the registry's 12-parameter `java_lcg_reverse_sieve`; `run_hybrid_reverse_sieve` is a bare `pass`. |
| `reverse_sieve_filter.py.BAK`, `reverse_sieve_filter.py.before_usage_docs`, `reverse_sieve_filter_FIXED.py` | Tracked backups of the above. |
| `coordinator_before_sieve_fix_20251011_114257.py`, `coordinator_sieve_dynamic.py`, `enable_sieve_dynamic.py`, `apply_sieve_cleanup_patch.py`, `fix_remote_sieve.patch`, `fix_reverse_sieve.patch` | Historical patch/backup artifacts, tracked. |
| `digit_sequential_sieve.py`, `per_draw_timestamp_sieve.py`, `pa_sieve_validation_harness.py`, `integration/sieve_integration.py` | Adjacent experiments, not the bidirectional sieve. |

---

## 8. Verification-integrity declaration (VIR-1…6)

**execution proof (VIR-1).** Every `file:line` in this map was obtained this session on VM 101
at `73dbacf`, by `Read` or by `/bin/grep -n` / `git show` against the working tree. No anchor
was carried over from a prior document without re-checking, with one deliberate exception:
anchors attributed to `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` in §2 are that
document's claims, validated **by diff** (`git diff --stat 42a7229..HEAD` shows the files they
point into are byte-unchanged) plus direct spot-verification of `prng_registry.py:958-1004`,
`:1005-1081`, `:1027` and all three `residues[::-1]` sites.

**clean control (VIR-2).** Per-source completeness is declared in §1 and tabulated in §9.
Sources 5 and 9 are **not** claimed as located; Source 4 carries a named unavailable leg. No
gap in this map is silent.

**fault-injection / positive control (VIR-2).** **Not applicable** — this pass ran no
detector, gate or harness. There is nothing here that could pass vacuously, because nothing
here executed. Stated rather than omitted, per the brief.

**unavailable-observer (VIR-5).** Nothing established by execution. Everything requiring a run
is marked UNAVAILABLE below rather than assumed.

**audit claim scope (VIR-6).**

*Searched surfaces:* the VM 101 working tree at `73dbacf` — `docs/*.md`, `miner/*.py`,
`prng_registry.py`, `sieve_filter.py`, `sieve_gpu_worker.py`, `coordinator.py`,
`window_optimizer_integration_final.py`, `utils/` (by grep), `tests/` (inventory + targeted
reads), root-level generators and duplicates; plus git history of
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` via `git log --follow` / `git show`. `/bin/grep` was
used throughout rather than the shell `grep` wrapper, so `.json` and gitignored files were
included in searches.

*Unavailable surfaces — declared, not assumed clean:*

1. **`window_optimizer.py`, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md`,
   `scripts/extract_search_bounds_snapshot.py`, `persistent_worker_coordinator.py`** — owned by
   the concurrent Chapter-1 P0 session; **not opened**. Affects Source 4 (PWC leg) and G-7.
2. **Deployed kernel source on the rigs.** All three CT100 workers are reachable — a ping
   sweep from VM 101 this session returned `192.168.3.122` UP, `192.168.3.156` UP,
   `192.168.3.164` UP — but **no comparison of the deployed `miner/`, `prng_registry.py` or
   kernel sources against VM 101 was performed.** Every kernel claim in this map is a claim
   about the VM 101 tree only. Repo ≠ system.
3. **Runtime values.** No GPU, sieve, miner, WATCHER or pipeline execution. No claim here
   rests on an observed run. Threshold, skip and partition behaviour is traced by source, not
   measured.
4. **`distributed_config.json` live values** were not re-read this session; where §1 needed a
   configured bound it cited the code that reads it, not a number.
5. **The 40 unreachable kernel sources** in `prng_registry.py` (non-java_lcg variants) were not
   read. Only `java_lcg`, `java_lcg_reverse`, `java_lcg_hybrid`, `java_lcg_hybrid_reverse` were
   opened.
6. **Team Beta's ruling texts** exist outside the repo except where transcribed; G-2 is
   precisely that observation.

---

## 9. Coverage table + completion sentinel

| # | Beta's derivation source | Authoritative live source | Coverage |
|---|---|---|---|
| 1 | Forward/reverse residue construction | `miner/range_miner_worker.py:538-580`, `:812-814`; `window_optimizer_integration_final.py:269-292`, `:1214` | **COMPLETE** |
| 2 | Host-side reverse ordering (`residues[::-1]`) | `miner/range_miner_worker.py:813`, `:116-117`; `sieve_filter.py:232`, `:395`; `sieve_gpu_worker.py:189`; `prng_registry.py:958-1004` vs `:3115-3169` | **COMPLETE** |
| 3 | Constant vs hybrid kernel semantics | `prng_registry.py:963/:972/:992`, `:1007-1012`, `:1022-1035`, `:1054`, `:1061-1077`; `miner/range_miner_worker.py:160-193`, `:776`, `:871`; `docs/HYBRID_SKIP_BOUND_AUDIT.md` | **COMPLETE** |
| 4 | Threshold and skip propagation | `window_optimizer_integration_final.py:210`, `:2363-2364`; `miner/range_miner_coordinator.py:1491-1523`, `:3134-3307`, `:3410-3430`; `miner/range_miner_worker.py:777-798`; audits cited | **COMPLETE** *(PWC leg unavailable — concurrency)* |
| 5 | Session-stream rules | `docs/DAILY3_CONSUMER_CONTRACT_v1.md` §4-§5, `:418`; `docs/TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md:34/:68/:188-195`; ruling text only in `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10 | **PARTIAL** — no ruling document (G-2) |
| 6 | Seed-domain partitioning | `miner/range_miner_coordinator.py:245-277`, `:311-381`, `:416-431`; `miner/range_miner_worker.py:472-503`; `coordinator.py:1415-1447` | **COMPLETE** |
| 7 | RANGE-MINER execution | `miner/range_miner_coordinator.py` (ledger/state machine/§6.8/retry), `miner/range_miner_worker.py` (lifecycle/ABI/transport), `miner/range_miner_protocol.py` | **COMPLETE** *(ABI contract undocumented outside code — G-4)* |
| 8 | D5/D6 assembly + provenance | `miner/assembly_backends.py:12-15/:45-51/:118-121/:273-284/:344-360`; `miner/range_miner_npz_writer.py:516-1268`; `miner/step1_ingress.py:143-265`; `utils/run_finalizer.py:690-827`; `utils/canonical_arrays.py`, `utils/canonical_records.py` | **COMPLETE** *(no as-built contract doc — G-6)* |
| 9 | Independent bounded known-answer controls (Wall C) | none — one-line spec at `docs/TFM_PROJECT_FACTS_SKILL.md:325`; raw materials only | **NOT FOUND** (G-3) |

**Additional deliverable items:**

| Item | Result |
|---|---|
| §2 `STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` assessment | **PARTIALLY USABLE** — high value on the legacy/PWC engine; §8.5/O7 superseded by `8a55a68`; miner absent by design; untracked, should be tracked |
| §3 Whitepaper-vs-chapter boundary | **PROPOSED** — mathematics stays in the whitepaper; the chapter states the implemented construction and flags the §4 `G(s,−i)` divergence without resolving it |
| §4 §14 disposition | **RETAIN**, line refs corrected and re-scoped to the legacy engine; **§15 SUPERSEDE** (duplicated verbatim; PWC non-certifying) |
| §5 Citation inventory | **COMPLETE** — 12 documents + one git object |
| §6 Gaps | **7 named** (G-1…G-7) |
| §7 Stale/duplicate inventory | **COMPLETE** — 7 documents, 12 source artifacts, none deleted |
| §0 Unplanned finding | **The lost 743-line chapter is recoverable at `d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`**; truncated by stale-copy overwrite at `248e48c` |

---

### Completion sentinel

```
STATUS:  INCOMPLETE
```

**`INCOMPLETE`, not `PASS`** — deliberately, and the reason is the deliverable, not a failure
of the pass. Seven of Beta's nine derivation sources are mapped COMPLETE to live source read
this session. **Source 5 is PARTIAL** (the binding ruling has no ruling document) and
**Source 9 is NOT FOUND** (Wall C has a one-line specification and no artifact; the one
convenient-looking candidate is semantically wrong). A source map that reported `PASS` while
two of nine sources have no authoritative repo location would be exactly the class of
"a check that was not checking, presenting as a pass" that VIR exists to prevent.

Sections 1-9 of the brief's §6 deliverable are all produced. Nothing in the tree was modified;
no commit was made.

**STOP — Team Alpha review gate.**
