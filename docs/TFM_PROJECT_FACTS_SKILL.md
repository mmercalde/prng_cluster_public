---
name: tfm-project-facts
description: Foundational model, verified as-built facts, superseded-artifact list, and mandatory verification procedure for Michael's TFM (Triangulated Functional Mimicry) distributed PRNG analysis project. Use this skill whenever the conversation touches TFM, the PRNG cluster, RANGE-MINER/S172, selfplay, Chapter 13/14, the bidirectional sieve, survivor pools, the NPZ contract, prediction pools, WATCHER, Zeus/VM101/rrig6600, or any file in the prng_cluster repos — even if the user does not name the project explicitly. Use before making ANY claim that something is missing, broken, unwired, unused, or current, and ALWAYS before proposing to remove, demote or simplify any component.
---

# TFM — Foundations, Verified Facts & Verification Procedure

**Currency:** through bounded Phase 6 certification (`d98298c`, 2026-08-02).

**§0 exists because of a specific failure.** In one session Team Alpha, Team Beta and Claude
Code *independently* recommended removing `skip_min`/`skip_max` from variable-skip search — a
cornerstone of the design — because no document any of them had read explained **why skip
exists**. All three inferred intent from current kernel signatures, which were the defect.
Michael stopped it. **This system has been in development over a year with the reasoning
documented and committed. If a component looks pointless, the explanation exists and has not
been read.**

---

## 0. FOUNDATIONS — read before proposing any change

### 0.1 The hypothesis
California Daily 3 draws are produced by a PRNG. Given observed history, find candidate
**seeds** whose generated sequences are consistent with what was drawn, then rank those
candidates with ML into prediction pools.
Basis: `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`.

### 0.2 Why bidirectional
Forward: seed `s` survives if match rate ≥ τ_f. For a random seed p = 1/1000 per position, so
survival decays exponentially in window length n. Reverse applies the same test on reversed
indices. For **incorrect** seeds the two are approximately independent:

```
P(survive both) ≈ P(survive forward)²      e^(−cn) → e^(−2cn)
```

**Bidirectional squares the exponent — a catastrophic collapse of noise.** At exact match with
n=50, false survival ≈ 10⁻³⁰⁰: only the true seed survives. That is why the sieve is
bidirectional.

**Reverse kernels iterate the PRNG FORWARD.** Direction comes from reversing residues on the
host (`residues[::-1]`); there is no inverse LCG — most PRNGs aren't invertible without full
state. **Do not flag this as a defect.**

### 0.3 Why thresholds are LOOSE — and must stay tunable
Whitepaper §7, counter-intuitive and misread every time:

> Exact sieves eliminate *all* variance. Survivors = {s*}. No ranking, no gradients, **no
> learning signal.**

Loose thresholds deliberately admit a *manifold* of near-consistent seeds sharing structured
deviations ML can learn to rank. **Loose thresholds are not sloppiness — they are a
mathematical necessity to expose learnable structure.** ML then works in a
survival-conditioned high-signal posterior, not raw PRNG space. This is why thresholds are
Optuna-tuned per direction, and why a threshold silently pinned to a constant is serious.

### 0.4 Why SKIP exists — the physical model
**The part nobody had written down.**

The published draw sequence is **not** an uninterrupted PRNG output stream. Per the
*California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09):

- **One automatic pre-test session runs before an automatic Daily draw** on the selected
  equipment (§V: Pre-Test via `[Start Draw Session]`). **Additional pre-test draws run only
  when an anomaly requires them.** Pre-test outputs are generated, verified, certified — and
  **never published.**
  *(Corrected 2026-08-01: previously "two pre-test draws run before every live draw" — an
  Alpha misreading. The "two test draws" language applies to **manual SuperLotto Plus
  equipment**. Only the count was wrong; skip remains physically motivated. Citation
  `UNAVAILABLE` — the PDF is not in the repo.)*
- **Draw equipment is selected per session** by an RNG program, auditor-verified (§II).
  Midday and evening are separate sessions with separate equipment selection.
- Evening draws **D3, D4, Fantasy 5 and Daily Derby together** — other games' outputs sit
  between the Daily 3 values you can see.

**Therefore the observable sequence has real, structural gaps of unknown and varying size.**
Skip models those gaps. It is a physical property of the data source, not a tuning convenience.

| mode | assumption | kernels |
|---|---|---|
| **constant skip** | fixed stride k between observed outputs | 22 kernels, all declare `int skip_min, int skip_max` |
| **variable skip (hybrid)** | stride varies, e.g. `[5,5,3,7,5,5,8,4,5,5]` | 22 kernels, declare `skip_sequences` + `strategy_tolerances` |

**Variable skip is a detector, not a fitting procedure.** It looks for windows where coherent
skip structure appears — the fingerprint glimpse — and produces survivors with *varied* skip
structure so tree/NN models have something to learn from. **The goal was never to reverse
state; it is to extract a fingerprint.**

**`skip_min`/`skip_max` ARE documented — in two readings, at two pipeline stages.** These are
**not contradictory**; they are the same names doing different jobs:

| stage | reading | source |
|---|---|---|
| **input** (Step 1 → 2) | *"Minimum/Maximum skip value **in pattern**"* — an **element-wise bound** on the discovered sequence. Documented hybrid default `[0,16]` | `docs/instructions.txt:1182-1183`; verbatim in `Cluster_operating_manual.txt:948-949`; present in an older revision, so it predates the current file |
| **output** (Step 2 → 3) | *"Minimum/Maximum gap that **worked**"* — an ML feature describing what the sieve found. *"Tight skip range = stronger hypothesis"* | `PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158`; `config_manifests/feature_registry.json:336,345` |

**Two registries currently disagree** — `feature_registry.json` says *"found during"* (output),
`parameter_registry.json:160,166` says *"for sieve search"* (input). One is wrong; correct it in
whatever change settles this.

**Mechanics that correct a common misreading:** `skip_sequences` is an **output, not an input**
(`prng_registry.py:1071`). **No pattern is generated.** `expected_skip = 5` seeds a *greedy
per-draw adaptive search that re-centres on each hit* (`:1047`), and the ancestor file still
carries `// Initial guess` (`prng_registry_pre_registry.py:696`) — a **guess, not a constant**.
`strategy_tolerances` is the half-width of the per-draw **matching** window
(`hybrid_strategy.py:20`), not generation. **No coherence scoring exists** — only `match_rate`.

*(The CA draw-procedures PDF is **not in the repo** — open item.)*

> **Standing rule.** Before recommending removal, demotion or simplification of ANY component,
> find and cite the document explaining why it exists. **Absence of a working implementation
> is not evidence of absent intent.** In a codebase this old, "the code doesn't do X" usually
> means X broke — not that X was never wanted.

### 0.5 Autonomy adjusts parameters, never structure
Whitepaper §9. The loop tightens thresholds over time; WATCHER/LLM governance proposes
*parameter* changes within governed bounds. It does not redesign the sieve.

**Corollary:** every tuned parameter must physically reach the kernel and its effective value
must be observable. A parameter the optimizer tunes but the kernel ignores is a **dead
dimension** — the sampler steers a knob connected to nothing and an autonomous agent would
"learn" into a void. This has now happened four times (§2.7).

### 0.6 The pipeline
```
Step 1  Window Optimizer (Optuna TPE)   → best window(s)
Step 2  Bidirectional Sieve             → survivors   [RANGE-MINER (S172) replaces this engine]
Step 3  Full Scoring (26 GPUs)          → scored survivors + 91-feature vector
Step 5  Model Training (4 families)     → best model (89 features) + diagnostics
Step 6  Prediction Generation           → pools (20/100/300)
Feedback  Chapter 13 daemon             → ingest draw → grade → (attribute) → decide → relearn
```
Carriers: the **22-array NPZ survivor contract** and the **prediction pool + coverage/lift
score**. Detail: `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (binding).

### 0.7 Why RANGE-MINER exists
PWC suffered silent hard resets / `GCVM_L2_PROTECTION_FAULT` on the RX 6600 rigs at full-fleet
saturation, traced to launch-storm behaviour. After weeks of failed debugging the project
pivoted to RANGE-MINER: **persistent per-GPU daemons**, standalone, producing **all** data the
remaining steps require — *the remaining steps must not be able to tell which engine produced
it.* That is an **interface** contract (the 22 arrays), not "match PWC's values."
**PWC is NOT the authoritative comparator** — Beta retired it from certifying authority
(2026-07-31); it is a flag-selectable, non-certifying diagnostic path.

---

## 1. THE RULE

**Never assert anything is missing, broken, unwired, unused, current or superseded without a
`file:line` anchor obtained in THIS session.** Unverifiable → label **[UNVERIFIED]**.

- **Trace the whole path**: producer → artifact → consumer. Real code with no producer is not
  "wired."
- **Check supersession.** 180+ sessions; existence ≠ current.
- **The repository is NOT the system (VIR-6).** systemd units, cron, host config, deployed
  uncommitted files are invisible to every repo gate. *(Alpha reported "no scraper invoker
  exists" from a clone; an enabled boot-triggered unit was on the host.)*
- **Read the consumer before specifying a change** — every producer, consumer, owner,
  lifecycle transition, path-type requirement, downstream schema dependency. For paths also:
  regular-file vs symlink ownership, compatibility aliases, atomic-replace boundaries,
  cleanup, concurrent-run namespace, restart consumers. *Three briefs in a row were defective
  for skipping this.*
- **§0.4's standing rule** — cite the design document before proposing removal.
- **A keyword hit is not a finding until the surrounding text is read.** *Four* absence claims
  were falsified in one session. The last — "nobody documented skip semantics" — was made after
  a full-tree grep **that reached the exact line and did not read it**
  (`HYBRID_SKIP_BOUND_AUDIT.md:318` vs `instructions.txt:1182`). Widening the search surface
  does not fix this; only reading the hits does.

---

## 2. SETTLED FACTS

### 2.1 Objective lineage
`score` → `holdout_hits` → **`holdout_quality`**. R² abandoned (0.000155 — zero signal);
**R² is not the objective.** Pipeline objective (S140b): `0.50·hit@20 + 0.30·hit@100 +
0.15·hit@300 + 0.05·pool_coverage`. `evaluate_pools.py` already computes coverage **and lift
vs random** — do not propose building it. Open: selfplay inner loop still uses R²;
`selfplay_orchestrator.py:933` prefers `holdout_hits`.

### 2.2 Feature contract
**91 extracted / 89 trained** (excl. `score`, `confidence`). "~62" is stale by 31. Three
namespaces: **72 survivor-local** (legitimate search space); **14 `global_*`** run-context
(identical for every survivor — filtering can only retain/remove the whole run; random row
folds leak run identity); **5 dead placeholders** with no producer (`skip_mean`, `skip_std`,
`skip_entropy`, `survivor_velocity`, `velocity_acceleration`).

**Three of those five are skip-shape statistics whose producer EXISTS on the GPU.** They are
dead because `extract_survivor_records` (`window_optimizer_integration_final.py:147`) discards
`skip_sequences`. The Oct-2025 output spec (`instructions.txt:1230-1245`) declares
`skip_pattern` and `pattern_stats: {mean_skip, variance, std_dev}` per survivor — the literal
ancestor of the three. **Reviving them requires no kernel change**, only that the host stop
discarding the sequence.

### 2.3 The 22-array NPZ contract (frozen)
Only **4 columns carry per-seed information**: `seeds`, `forward_matches`, `reverse_matches`,
`score`. **RANGE-MINER emits exactly 22 arrays, nothing more.**
`forward_matches`/`reverse_matches` are the only independent per-seed sieve signal and are
**absent from the Step-3 merge list** — TB: possibly the most consequential finding in the
trace. Needs a governed schema decision; the miner keeps emitting both regardless.

### 2.4 Attribution
`per_survivor_attribution.py` is real and invoked with seed identity — **implemented, invoked,
unreachable, unconsumed.** Four blockers. Never say "wired" or "not implemented"; both wrong.

### 2.5 Selfplay
A **policy-conditioned evaluation harness**, not a learning system.
`propose_transform_update` is a no-op; promotion seam broken
(`chapter_13_acceptance.py:224`). All Ch13 triggers **defensive**; no opportunity trigger.

### 2.6 Looks-like-a-bug, isn't
Reverse = host-side residue reversal (§0.2). Loose thresholds required (§0.3).
`intersection_count` duplicating `bidirectional_count` is deliberate.
**`serve_timeout=None` is deliberate** — a billion-seed scan exceeds any wall clock; the
bounded clock is on *admission* only (§2.12). **`distributed_config.json`'s bare-metal
addresses are deliberate** — both topologies are retained as profiles (§2.11). **`run_optimization`'s `sampler` and `sampler_metadata` are
REQUIRED and keyword-only with no default** — deliberately, so a caller cannot get TPE by
omission and then report the run as something else; an unlabelled run is not a control.
**Chain D's `pending_approval`** is a valid authority boundary and the **Step-5 `allowed_params` filter**
is a deliberate executable-interface boundary (§2.13).

### 2.7 Recurring defect: tuned parameters don't reach kernels — SIX instances

| # | instance | status |
|---|---|---|
| 1 | miner filtered at hardcoded `0.25` | **FIXED** `2be51d5` — single canonical path, per-direction resolution in the parent, effective value read off the executor, parent-side fail-closed provenance |
| 2 | Optuna thresholds dropped above `run_bidirectional_test`; every trial ran `0.30/0.30` | **FIXED** `8a55a68`. Was a **regression**: fixed `3fdf434` (04-30), silently reverted `2389b61` (07-07) by a stale-copy overwrite whose message never mentions thresholds. Both routes now use `resolve_directional_threshold()`, `is None` not truthiness (**0.0 is legitimate**) |
| 3 | PWC hybrid filtered at `0.50` | **QUARANTINED** — `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; PWC non-certifying, so the defect is made loud rather than repaired |
| 4 | **hybrid kernels ignore sampled `skip_min`/`skip_max`; `expected_skip = 5` hardcoded** | **OPEN.** 22/22 constant kernels declare skip bounds; 0/22 hybrid do. Values survive eight hops and **die at `_hybrid_prefix`** (`range_miner_worker.py:177-193`). Anchors `prng_registry.py:1027, :805, :885, :1159`. **Semantics ARE documented (§0.4)** — the "unspecified semantics" premise of `HYBRID_SKIP_BOUND_AUDIT.md:318` is **FALSE**. Decision open; **the output-statistic reading needs no kernel change at all** |
| 5 | forward hybrids ignore `offset` (sampled `window_optimizer_bayesian.py:423`) | **OPEN** |
| 6 | **`skip_learning_rate`** configured 0.2–0.7; kernel **hard-adapts at 1.0** | **OPEN**, newly catalogued |

Fix pattern: **one canonical path** — resolve once in the parent, never reinterpret
downstream, record requested/payload/effective.

### 2.8 RANGE-MINER Phase 5 as-built (committed, dual-pushed)
**D3.5** shared run finalizer, immutable chain-authenticated generations; owns the root
compatibility **symlinks** and **fails closed if a regular file appears there**.
**D4** `serial_reference` behind a frozen two-backend interface that fails closed.
**D5** `process_sharded` parallelizes **only** spool-local validation; parent alone owns
merge/dedup/intersection. **Available, UNPROMOTED** (~1.6× faster high-survivor at ~2–3× RAM;
~180× slower low-survivor).
**D6** production adapter; miner candidates reach the Step-1 accumulator and a certified
generation; per-direction thresholds reach the kernel with requested/payload/effective
provenance and parent-side fail-closed enforcement; shared session-filtered residue authority.
**D6.1** incremental checkpoint **writes for the first time** — run-isolated
`.s172_checkpoint/<run_id>/`, open-handle write + fsync, six-field transaction identity.
**NON-AUTHORITATIVE four-field snapshot.**
**Phase 6.0** RX 6600 XT (ROCm) and RTX 3080 Ti (CUDA) produce a **byte-identical certified
artifact** — same `artifact_sha256` across the D6 release-grade generation and both 6.0 runs;
22/22 arrays equal; **no GPU reset, no `GCVM_L2_PROTECTION_FAULT`** in the host kernel log.

**Authoritative artifact:** `gen-20260730T002104136270Z-step1_java_lcg_0`, commit `b08c2c5`,
`artifact_sha256 0e0092fe…c4b0` — the **pre-dataset-provenance authoritative generation**.
The ROCm generation is **platform-validation, non-authoritative**.

**Phase 6-P0** `131787d` · **6-P0.5** `d4ff1e4` · **P0.5 Q2 closure** `8600e75` — dataset
authority, see §2.10. **§4.3 admission liveness** `ee0db06` — see §2.12.

**Bounded Phase 6 — CERTIFIED and CLOSED** `d98298c` (TB ruling, 2026-08-02). Wall A: the
complete consumer chain — frozen 22-array bundle → validation → Step-2 load without fallback →
dict conversion → Step-3 chunks → real GPU scorer, with **value-by-value** metadata comparison
(closing the "keys present but values defaulted" class). Wall B: repetition, assembly-backend
equivalence, current CUDA/ROCm equivalence, and **node-assignment independence across two
different ROCm rig pairs** — all five arms reproduced `0e0092fe…c4b0`. **Miner Known-Answer
Transfer Gate:** all four active TFM variants through their real `SieveExecutor.execute` ABI
paths; **eight populations exact-set equal, zero missing / extra / mismatched**; F5–F7 prove
reference independence by rejecting three wrong semantics.

**Certification scope is explicit:** Wall A/B used **constant-skip** generations; **hybrid worker
semantics are covered by the transfer gate, not by a full four-phase Wall-A consumer run.**
The scratch generations are **not** release-grade — future publication still uses
`--release-grade`.

### 2.9 Known-disabled / deliberately off
**S166 in-memory clear** — disabled; candidate-list RAM growth unbounded until the checkpoint
carries all 24 `CANONICAL_RECORD_FIELDS` (carries 4). **D6.2, Phase-7 blocker.**
**`.s172_checkpoint/<run_id>/` never pruned** — **D6.3, Phase-7 blocker.**
**`process_sharded`** selectable, unpromoted.
**`daily3scraper.service`** — enabled since Sep 2025 with `Restart=always`, target
`run_daily3scraper.py` **never existed**; ENOENT loop every boot. **Now `disable --now`, unit
retained.** Stays disabled until Phase 6-P2 is certified.
**PWC/ZMQ** retired from certifying authority; PWC hybrid additionally quarantined.
**`dataset_provenance/*.json` never pruned** — same class as D6.3, newly found.
**Sampler provenance is unverified** — `run_optimization()` trusts caller-supplied
`sampler_class` / `sampler_module` / `optuna_version` and does not check them against the actual
object. Existing TPE and Random wrappers are correctly labelled, so nothing submitted is
invalidated; **a fail-before-study guard is required before direct use of the neutral core or
registration of another sampler.**
**`process_sharded` import invariant has no gate** — TB **REQUIRES** one: fresh spawned
interpreter · real Step-1 module surface · invoke the **production**
`assembly_shard_worker.assert_cpu_only()` (do not duplicate its forbidden list) · cover **both**
`torch` and `cupy` · plus a mutant introducing a module-level GPU import that proves it reds.
Required hardening, **not** a Phase 6 blocker — the real `process_sharded` arm passed.
**`quick_test_all_22.sh`** is **differential/liveness evidence only, never known-answer
correctness evidence** (TB). Its output path is now timestamped so the supersession record
survives.
**`random`/`grid`/`evolutionary` strategies** gated at the CLI (signature mismatch) — **not
deleted**. Documented design was four Optuna samplers; `GridSampler` is **unconstructible**
here (7.649 × 10¹⁰ grid points ≈ 7.2 TiB at construction).

### 2.10 Dataset authority — LIVE as of P0.5
- **The pointer manifest is authoritative.** `daily3_current.json` → immutable
  `daily3-<UTC>Z-<sha256[:12]>.json`. **`daily3.json` is now a legacy compatibility alias.**
- **Resolved ONCE at run start and frozen** — manifest identity, absolute path, sha256, size,
  record count. A pointer moving mid-run **cannot alter a run in progress**. `dataset_sha256`
  moved from **per-trial to run scope** (`range_miner_coordinator.py:85`); a mid-study scrape
  used to split a study across two datasets with **no error anywhere**.
- **Dispatch is the absolute immutable path**, never the bare alias. **Fail before first worker
  dispatch.** All three rigs provisioned through one path, digests verified **on target**.
- `DatasetProvisioningError(ResidueError)` — chained, names absolute path **and node**.
- **The `.json` extension is load-bearing**: `.gitignore:41` keeps published artifacts out of
  the clean-tree check at `run_finalizer.py:1589`. **No sidecars** — the digest lives inside the
  manifest. Do not name it `*_config.json` / `schema_*.json`.
- **A local single-GPU run now refuses while any rig is down** — the fail-closed reading;
  a bypass needs governing.
- `dataset_provisioning.json` is gitignored — a fresh clone has no fleet definition.
- **A miner-backed run HARD-FAILS** on a **missing, unreadable, invalid or empty** provisioning
  manifest, before any coordinator construction or dispatch (`8600e75`). Unreadable and invalid
  are decided **in the loader** — a manifest nobody can read establishes nothing for anyone —
  and only *absent* is a caller decision.
- **Status vocabulary, and it is load-bearing:**
  **`UNAVAILABLE`** = a required verification was **attempted and could not complete** → fatal
  for a miner topology. **`NOT_APPLICABLE`** = this path never needed the check → proceed.
  *"We needed it and could not get it" is not "we did not need it."* An **unknown**
  `remote_execution` keeps the over-constrained reading (`UNAVAILABLE`), never the clean one.
- **`remote_execution=False` is a topology statement, NOT a bypass.** It must never become
  Beta's Q1 refinement by the back door: a local run that still drives the 26-GPU coordinator
  **performs remote execution** and must not declare otherwise.

### 2.10b Dataset lifecycle (TB rulings 2026-07-30/31)
- Midday and evening use **independently selected equipment** — **no evidentiary basis for
  advancing one PRNG state through interleaved records.** Ordering is normative **within a
  session stream**; combined-container order carries **no PRNG-advance meaning**. The
  chronological-reorder migration was **cancelled**. Combined-session sequential sieve is
  **non-certifying, prohibited by default**; production re-optimization is **per-session**.
- Scraper moves to **append-only immutable versioned files** + an **atomic pointer manifest**
  (not a bare symlink). Version IDs need UTC timestamp **and** content identity.
- **Two walls:** *publication prefix* (history not rewritten — a **record-sequence** check; a
  byte-prefix test is invalid for JSON arrays) and *accumulator input* (**exact
  input-manifest digest match**). **Append-only does NOT make prior scores valid on the next
  version** — adding a draw changes windows, eligibility, gap/skip features, global frequency,
  normalization, any "latest N". Prefix-only merging **not approved**.
- Corrections → **new dataset lineage**, old preserved; scraper stops with
  `CORRECTION_REQUIRED` and may not create the corrected lineage autonomously.
- Generations **chain** (the finalizer merges prior rows) — input identity is a lineage
  invariant, not annotation.

---

### 2.11 Fleet authority — six mechanisms, no single definition (TB fleet ruling)

**There is no single required fleet state today.** Six checks at three granularities on two
address sets; which apply depends on the backend flag and whether the run came via WATCHER or
the CLI.

| mechanism | granularity | addresses |
|---|---|---|
| P0.5 dataset preflight | node | **`.122/.156/.164`** (CT100) |
| legacy `test_connectivity` (`coordinator.py:502`) | node | `.120/.154/.162` |
| PWC ready gate (`persistent_worker_coordinator.py:864`) | **GPU** | `.120/.154/.162` |
| WATCHER GPU health (`preflight_check.py:293`) | GPU | `.120/.154/.162` — **non-blocking by design**, WATCHER-only |
| boot notify | GPU | host-local, **Telegram-only, `exit 0`** |
| miner `expected_workers` | worker daemon | **whoever connects** |

**Three** point at bare metal; P0.5 points at the CT100s; two name no fixed set. The rigs are
booted into Proxmox, so **P0.5 passes and the three bare-metal checks structurally cannot** —
P0.5 is the only mechanism updated for the migration.

**Beta's ruling: none of the six defines the fleet.** The future sole authority is a **frozen,
run-scoped Resolved Execution Set**, created after backend and rig-profile selection but
**before** dataset verification, GPU verification, coordinator construction or dispatch. It
carries backend · rig profile · logical nodes and endpoints · worker/GPU identities ·
local-vs-remote · admission count · dataset-verification targets. **WATCHER and the CLI invoke
the same resolver.** All six become **consumers**. **A partial set must be explicit and frozen
before the run — never inferred from which workers happened to answer**, and unknown miner
workers must not become eligible merely because they connected.

**Both topologies are retained** (TB ruling 3): `.120/.154/.162` is the deliberate bare-metal
profile, `.122/.156/.164` the Proxmox compute endpoints. **The selected profile decides which
endpoints enter the set.** `distributed_config.json`'s bare-metal addresses remain
deliberate — see §4.

**Q1 — local runs.** A local single-GPU run currently **refuses while any rig is down**. The
refinement (verify only the resolved execution set) is approved **in principle** but must come
through the shared resolver, **not** by special-casing P0.5 or weakening `require_fleet`. Until
then the over-constrained behaviour stands.

### 2.12 Admission liveness — the §4.3 hang, repaired `ee0db06`

**The defect:** `assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and** the stage
advance were all behind one guard, `len(eligible) >= expected_workers` — and `serve_timeout` is
`None` by design. A worker loss crossing the threshold stopped lease expiry from being
processed, so dead workers' stripes stayed `claimed` with expired leases nobody looked at:
**the trial neither completed nor failed.** *The Blocker-3 matrix was unreachable in exactly
the situation it exists for.*

**The repair separates admission from maintenance:**
- **ADMISSION (bounded)** — reaching `expected_workers` is a precondition for *assigning* a
  stage, bounded by **`worker_admission_timeout`, default 180 s** (the PWC readiness window).
  Failure is an explicit `fail_trial` naming run · stage · family · phase · expected · admitted
  · elapsed. **The window re-arms only when `stage_idx` changes**, so worker churn cannot
  extend it.
- **MAINTENANCE (unbounded)** — once a stage is assigned, dispatch, lease expiry and completion
  evaluation run **regardless of the current eligible count**. A shrunken or empty pool is a
  legitimate input the matrix already handles.

**`serve_timeout` stays `None`** — a multi-billion-seed scan exceeds any wall clock, and the
bounded clock is on *admission only*. `expected_workers` is **not** reduced dynamically;
`worker_pool_size` semantics and the Blocker-3 matrix are **unchanged** — the matrix is now
*reachable*, not rewritten.

### 2.13 Control chains, end to end
*Which knobs actually reach execution. This table exists so a wiring gap is found now, not at
Chapter 13.*

| chain | emit → validate → apply → execute | state |
|---|---|---|
| per-direction thresholds → kernel | ✅ ✅ ✅ ✅ | **WORKS** (D6 + `8a55a68`) |
| dataset identity → all nodes | ✅ ✅ ✅ ✅ | **WORKS** (P0.5 + Q2 closure) |
| fleet definition → the run | ✗ — — — | **six mechanisms, no authority** (§2.11) — Resolved Execution Set is the approved fix, unbuilt |
| worker loss → failure matrix | ✅ ✅ ✅ ✅ | **WORKS** (`ee0db06`) — was an unbounded hang |
| Advisor → selfplay `max_episodes`, `min_fitness_threshold` | ✅ ✅ ✅ ✅ | **WORKS** |
| Optuna `skip_min`/`skip_max` → hybrid kernel | ✅ ✅ ✅ ✗ | dies at `_hybrid_prefix`. **The approved skip-OUTPUT work does NOT fix this** — see §8 |
| miner kernels → independent reference | ✅ ✅ ✅ ✅ | **WORKS** — transfer gate, 8/8 populations exact-set equal (`d98298c`) |
| Optuna `offset` → forward hybrid | ✅ ✅ ✅ ✗ | dies in kernel args |
| `skip_learning_rate` → kernel | ✅ — — ✗ | kernel hard-adapts at 1.0 |
| Advisor → `search_strategy` | ✅ partial ✗ — | dies in the override dict. **Autonomous application NOT approved** |
| Advisor → `strategy_recommendation.json` → WATCHER | ✅ ✅ ✗ — | **no code reads the file**; the working path is in-memory |
| diagnostics → Step-5 retry params | ✅ ✅ ✅ ✗ | filtered at the step boundary — **deliberate**; reporting fixed `f8b751c` |
| Ch13 proposal → acceptance | ✅ ✅ ✗ — | `pending_approval` is a **valid authority boundary** |
| GPU `skip_sequences` → ML features | ✅ — ✗ — | discarded at `…final.py:147`; kills 3 features |

**Reserved authority (human only):** feature engineering · survivor thresholds · sieve
strategy/mathematics · window-optimizer logic · PRNG-family authority · scoring logic ·
meta-optimizer search space · model families · policy authority.

## 3. SUPERSEDED — in repo, NOT current
R² as objective · `holdout_hits` as ML target · `feature_importance.py` 60-name list (stale by
31) · "~62 features" · `bidirectional_survivors.json` as survivor data ·
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (fragment) · `run_full_scoring.sh` · **PWC/ZMQ as
certifying comparators** · `full_scoring_worker.py` "50 features" · `window_optimizer.py:450`
docstring · `RUNTIME_DATASET_PROVISIONING_CONTRACT.md` `expected_sha256` as static config ·
scraper `--rewrite` mode · "RX 6600" on rrig6600 (they are **6600 XT**, 32 CUs — inventory per
node) · "the writer is unconditionally frozen" (D6 added one approved `backend=None` seam).

## 4. FROZEN — reuse, never reimplement
- **`_l2_sort_key` / `_select_l2_winners`** (`utils/run_finalizer.py:690`, `:714`, Ruling D):
  highest **float32** score → lowest `trial_number` → constant-before-variable *within a trial
  only*; same-trial/same-mode collision raises `AccumulatorConsistencyError`. Comparing
  pre-rounding float64 **is the defect this converts away.** Import; never fork.
- **D3.5 finalizer-owned root symlinks** — a regular file there makes `finalize_run` raise
  `PublicationError`.
- **D5 §6.7.A compressed-artifact ban** — scoped to worker *transport* artifacts. The D6.1
  **checkpoint may be compressed**; deliberately separate. Do not harmonize.
- **The 22-array NPZ contract**; the D3.25 four-map ingress contract.
- **`distributed_config.json` bare-metal addresses** (`.120/.154/.162`) — deliberate, they
  match the *default boot target*. `CLAUDE.md` §3: **not a bug, must not be corrected.**

## 5. VERIFICATION INTEGRITY (VIR-1…6) — binding
`docs/VERIFICATION_INTEGRITY_STANDARD.md`. Adopted after three incidents of *a check that was
not checking, presenting as a pass*.
**VIR-1** verification must prove its own execution; silence/truncation/reporter death/an
inaccessible surface is never a pass. **VIR-2** vacuous-capable detectors need execution proof
· **clean control** · **fault-injection (positive) control** · detector independence (*not
interchangeable terms*). **VIR-3** terminate in `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only
`PASS` accepts. **VIR-4** cleanup must not kill its reporter. **VIR-5** unobservable is not
clean. **VIR-6** audit scope must match the claim; declare searched **and** unavailable
surfaces.

Gates should extract and execute the **live source** of the call site (AST), not match text —
`2389b61` reverted a fix by whole-block replacement; a text anchor would have gone green.

Every brief carries:
```
Verification-integrity controls (VIR-1…6):
- execution proof:      - clean control:      - fault-injection control:
- completion sentinel:  - unavailable-observer behavior:
- audit claim scope:    - searched surfaces:  - unavailable surfaces:
```

## 6. TOPOLOGY (verified 2026-08-01)
Rigs boot **bare-metal by default**; currently in **Proxmox**. `host = rig+1`,
`CT100 worker = host+1`.

| rig | bare-metal | Proxmox host | **CT100 worker (use this)** |
|---|---|---|---|
| rrig6600 | `.120` | `.121` | **`192.168.3.122`** |
| rrig6600b | `.154` | `.155` | **`192.168.3.156`** |
| rrig6600c | `.162` | `.163` | **`192.168.3.164`** |

Key auth works **from VM101**, not ser8. All three CTs: `~/rocm_env`, **8 × RX 6600 XT**, cupy
13.5.1, gfx1032, **no HSA/GFX overrides needed**. **Venvs differ** — VM101 `~/venvs/torch`,
rigs `~/rocm_env`. CT100 is an **unprivileged LXC**: GPU kernel log must be read from the
Proxmox host (`root@.121`). `daily3.json` is **gitignored** — clone alone can't stand up a rig.
**All three rigs are provisioned with the frozen dataset** (P0.5, verified on target).
`dataset_provisioning.json` is also gitignored — a fresh clone has no fleet definition.

## 7. WORKING AGREEMENTS
- **Claude = Team Alpha** (lead dev). **Team Beta** = separate approval authority; rulings
  binding; never impersonate. **Alpha may contest with evidence** — Ruling 20 was withdrawn
  that way — but does not overrule.
- **Never commit or push from the sandbox.** Deliver to `/mnt/user-data/outputs/`; Michael
  downloads to ser8, `scp`s to VM101, commits, dual-pushes.
- **EVERY command must name its host.** ser8 = download target and `scp` source. VM101 = repo,
  all `git`, all rig SSH. *The single most repeated operational failure.*
- **`source ~/venvs/torch/bin/activate` before any test command.** A bare shell yields
  `CuPy not available` / `Optuna not available` — false reds. Watch for `(torch)`.
- **Plans for behaviour-changing work go to Beta before implementation.** P0's procedural
  exception was granted because it was inert; Beta stated it is **not precedent**.
- **Stage explicitly, never `git add -a`.**
- **Never reuse a filename when uploading to chat.** Same-named uploads deduplicate somewhere
  upstream and arrive as **empty stubs** — six transfers were lost this way in one session
  before the cause was found. Give every upload a unique name. `.diff` and `.png` were 100%
  reliable; repeated `.txt`/`.md` names were not. **The repository is the fallback channel**:
  commit and push, then the reader clones — that path has never failed.
- Session changelog to `docs/`; dual-push `origin` (private) + `public` (mirror) —
  **everything pushed is effectively public.**
- Prefer **Claude Code on VM101** for as-built questions — live source **and** live host. A
  clone is repo-only (VIR-6); chat-side reasoning is provisional.
- Briefs: one falsifiable question, a defined deliverable, "write the report to
  `docs/<n>.md`". Separate *investigate* from *fix*.
- Long nested suites look hung under `| tail`. Check for a **descendant** process before
  concluding a hang; a blocked parent burns no CPU.

## 8. APPROVED SEQUENCE
```
D6.1 ✅ · Phase 6.0 ✅ · threshold repair ✅ · Ch1 P0+P1/P2 ✅ · Chain C ✅
6-P0 ✅ · 6-P0.5 ✅ · Q2 closure ✅ · §4.3 liveness ✅ · Wall C struck ✅
bounded Phase 6 ✅ CERTIFIED and CLOSED d98298c
next   Resolved Execution Set + profile-aware fleet consumers (§2.11)
       process_sharded import gate (TB-required, §2.9)
       skip-OUTPUT work · D6.2 · D6.3 · 6-P2 scraper
Phase 7  BLOCKED by the Resolved Execution Set, D6.2, and the other pre-Phase-7 obligations
```

**⚠ Sampler-comparison sequencing — a correction TB issued against Alpha.** The certifying
four-phase TPE-vs-random comparison **cannot** be scheduled merely *"after the skip-output
work."* The approved skip-output work retains observed sequences and restores `skip_mean` /
`skip_std` / `skip_entropy`. **It does NOT connect `skip_min`/`skip_max` to the hybrid kernels**
— that is the separate, unresolved **input-bound** interpretation. The comparison must wait
until **either** hybrid search-input bounds have defined effective semantics, **or** the
comparison uses an **explicitly phase-aware search space that does not pretend dead hybrid
dimensions are active.** Skip-output may proceed first; completing it alone **does not remove
the dead-dimension caveat.**

**TPE remains the production default by status quo** — the five-seed run is a valid constant-skip
datapoint and useful directional evidence, **not** a certification of superiority and **not**
authority for autonomous sampler selection.

**Open backlog:** Chapter 2 restore-and-audit (recoverable at `d14dcdd`) · Chapters 3–7 audits ·
three `[WATCHER][RETRY]` log lines with the Chain C defect · two doc-generator defects ·
**session-separated dataset authority** · `.gitignore:42` dead negation · the CA
draw-procedures PDF is not in the repo · the `java_lcg_cpu` non-zero-skip mismatch at
`survivor_scorer.py:124` / `full_scoring_worker.py:305` (TB: separate bounded audit before
Phase 7, **no fix authorized**).
**Wall C caution:** `java_lcg_cpu` (`prng_registry.py:170-183`) applies skip **once before
generating**; the kernel applies it **between every draw** (`:987-989`). They agree only at
`skip=0`. **Building the known-answer reference on it would validate the wrong semantics** in
the deliverable meant to catch semantic error. *(Michael reports all 44 PRNGs were validated
through the sieves during pipeline development — constant forward/reverse and hybrid
variable-skip. An inventory is establishing what exists before anything is scoped as new.)*

**Open backlog:** Chapter 2 restore-and-audit (recoverable at `d14dcdd`) · Chapters 3–7 audits ·
three `[WATCHER][RETRY]` log lines with the Chain C defect · two doc-generator defects ·
**session-separated dataset authority** · `.gitignore:42` dead negation · the CA
draw-procedures PDF is not in the repo.
**Open backlog:** Chapter 2 restore-and-audit (recoverable at `d14dcdd`) · Chapters 3–7 audits ·
three `[WATCHER][RETRY]` log lines with the Chain C defect · two doc-generator defects ·
**session-separated dataset authority** · `.gitignore:42` dead negation · the CA
draw-procedures PDF is not in the repo.
**Hard Phase-7 prerequisites:** 6-P0, 6-P1, D6.2, D6.3, 6-P2.
**Optuna:** constant-skip **may resume**; hybrid exploration **non-certifying only**; hybrid
certification **blocked** until skip bounds are live; authoritative studies need provenance
binding.

## 9. SELF-CHECK BEFORE SENDING
1. Claimed something missing/broken/unwired/current? → anchor or **[UNVERIFIED]**.
2. **Proposing to remove, demote or simplify anything? → did I cite the doc explaining why it
   exists (§0.4)?**
3. **Did I READ the grep hits, or only count them?** (§1, fifth corollary)
3. Cited a metric or target? → check §3.
4. Proposing something that already exists? (coverage metric, downstream_score, attribution
   engine).
5. Classified a capability from one module without tracing producer → artifact → consumer?
6. Changing a shared buffer, path or format? → enumerated every consumer?
7. System-scoped claim on repo-scoped evidence? (VIR-6)
8. **Named the host for every command? Included the venv activation?**
9. Long thread? Verification discipline degrades — suggest a fresh session.
