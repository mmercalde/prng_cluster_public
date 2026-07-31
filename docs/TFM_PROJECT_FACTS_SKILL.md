---
name: tfm-project-facts
description: Foundational model, verified as-built facts, superseded-artifact list, and mandatory verification procedure for Michael's TFM (Triangulated Functional Mimicry) distributed PRNG analysis project. Use this skill whenever the conversation touches TFM, the PRNG cluster, RANGE-MINER/S172, selfplay, Chapter 13/14, the bidirectional sieve, survivor pools, the NPZ contract, prediction pools, WATCHER, Zeus/VM101/rrig6600, or any file in the prng_cluster repos — even if the user does not name the project explicitly. Use before making ANY claim that something is missing, broken, unwired, unused, or current, and ALWAYS before proposing to remove, demote or simplify any component.
---

# TFM — Foundations, Verified Facts & Verification Procedure

**Currency:** through the S172 threshold repair (`8a55a68`, 2026-07-31).

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

- **Two pre-test draws run before every live draw** on the selected equipment (§V: Pre-Test
  via `[Start Draw Session]`; *"Run Draw as Test"* is unchecked only afterwards). Pre-test
  outputs are generated, verified, certified — and **never published.**
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

`docs/CHAPTER_1_WINDOW_OPTIMIZER.md` defines the fields explicitly: **`skip_min` = "Minimum
skip for variable PRNGs"**, **`skip_max` = "Maximum skip for variable PRNGs"**; search space
`skip_min` 0–10, `skip_max` 10–500. The fields are documented **for the variable case**. The
current hybrid kernels not accepting them is **the defect**, not the design.

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

### 2.7 Recurring defect: tuned parameters don't reach kernels — FOUR instances

| # | instance | status |
|---|---|---|
| 1 | miner filtered at hardcoded `0.25` | **FIXED** `2be51d5` — single canonical path, per-direction resolution in the parent, effective value read off the executor, parent-side fail-closed provenance |
| 2 | Optuna thresholds dropped above `run_bidirectional_test`; every trial ran `0.30/0.30` | **FIXED** `8a55a68`. Was a **regression**: fixed `3fdf434` (04-30), silently reverted `2389b61` (07-07) by a stale-copy overwrite whose message never mentions thresholds. Both routes now use `resolve_directional_threshold()`, `is None` not truthiness (**0.0 is legitimate**) |
| 3 | PWC hybrid filtered at `0.50` | **QUARANTINED** — `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; PWC non-certifying, so the defect is made loud rather than repaired |
| 4 | **hybrid kernels ignore sampled `skip_min`/`skip_max`; hardcoded `expected_skip = 5`** | **OPEN — next task.** 22/22 constant kernels declare skip bounds; 0/22 hybrid do. Live on the **certifying miner route**: `range_miner_worker.py` reads `skip_range` (`:776`) into `BuildContext` (`:871`), and `_hybrid_prefix` (`:177-193`) never emits it — values survive argparse, config, coordinator, ledger, manifest, payload, worker unpack and the arg-build context, then **die one call before launch.** Anchors `prng_registry.py:1027, :805, :885, :1159`. **Fix by wiring in, NOT removal (§0.4).** Forward hybrids ignore `offset` too (sampled `window_optimizer_bayesian.py:423`) — same class |

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

### 2.9 Known-disabled / deliberately off
**S166 in-memory clear** — disabled; candidate-list RAM growth unbounded until the checkpoint
carries all 24 `CANONICAL_RECORD_FIELDS` (carries 4). **D6.2, Phase-7 blocker.**
**`.s172_checkpoint/<run_id>/` never pruned** — **D6.3, Phase-7 blocker.**
**`process_sharded`** selectable, unpromoted.
**`daily3scraper.service`** — enabled since Sep 2025 with `Restart=always`, target
`run_daily3scraper.py` **never existed**; ENOENT loop every boot. **Now `disable --now`, unit
retained.** Stays disabled until Phase 6-P2 is certified.
**PWC/ZMQ** retired from certifying authority; PWC hybrid additionally quarantined.

### 2.10 Dataset lifecycle (TB rulings 2026-07-30/31)
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

## 6. TOPOLOGY (verified 2026-07-30)
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

## 7. WORKING AGREEMENTS
- **Claude = Team Alpha** (lead dev). **Team Beta** = separate approval authority; rulings
  binding; never impersonate. **Alpha may contest with evidence** — Ruling 20 was withdrawn
  that way — but does not overrule.
- **Never commit or push from the sandbox.** Deliver to `/mnt/user-data/outputs/`; Michael
  downloads to ser8, `scp`s to VM101, commits, dual-pushes.
- **Every command must name its host.** ser8 = download target and `scp` source. VM101 = repo,
  all `git`, all rig SSH.
- **Stage explicitly, never `git add -a`.**
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
D6.1 ✅ · Phase 6.0 ✅ · threshold repair ✅
next    hybrid skip-bound dead dimension (§2.7 #4) — WIRE IN, do not remove
        study↔commit/dataset provenance binding
6-P0    freeze publication/pointer/correction schemas; bootstrap publication;
        pointer resolution, fleet provisioning, fail-before-dispatch
6-P1    dataset provenance binding + exact-input accumulator wall
bounded Phase 6 — three walls: (A) interface/consumer incl. Step-3 · (B) determinism/platform
        · (C) **bounded independent known-answer correctness** — a reference that does NOT call
        the miner's coordinator/backend/finalizer
D6.2 · D6.3 · 6-P2 (order flexible)
Phase 7  26-GPU saturation + WATCHER soak
```
**Hard Phase-7 prerequisites:** 6-P0, 6-P1, D6.2, D6.3, 6-P2.
**Optuna:** constant-skip **may resume**; hybrid exploration **non-certifying only**; hybrid
certification **blocked** until skip bounds are live; authoritative studies need provenance
binding.

## 9. SELF-CHECK BEFORE SENDING
1. Claimed something missing/broken/unwired/current? → anchor or **[UNVERIFIED]**.
2. **Proposing to remove, demote or simplify anything? → did I cite the doc explaining why it
   exists (§0.4)?**
3. Cited a metric or target? → check §3.
4. Proposing something that already exists? (coverage metric, downstream_score, attribution
   engine).
5. Classified a capability from one module without tracing producer → artifact → consumer?
6. Changing a shared buffer, path or format? → enumerated every consumer?
7. System-scoped claim on repo-scoped evidence? (VIR-6)
8. Named the host for every command?
9. Long thread? Verification discipline degrades — suggest a fresh session.
