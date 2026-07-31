---
name: tfm-project-facts
description: Verified as-built facts, superseded-artifact list, and mandatory verification procedure for Michael's TFM (Triangulated Functional Mimicry) distributed PRNG analysis project. Use this skill whenever the conversation touches TFM, the PRNG cluster, RANGE-MINER/S172, selfplay, Chapter 13/14, the bidirectional sieve, survivor pools, the NPZ contract, prediction pools, WATCHER, Zeus/VM101/rrig6600, or any file in the prng_cluster repos — even if the user does not name the project explicitly. Also use before making ANY claim that something in this project is missing, broken, unwired, unused, or current.
---

# TFM Project — Verified Facts & Verification Procedure

This skill exists because of a specific, repeated failure: **asserting as-built facts from
partial reads instead of verifying them.** In one session, five separate confident claims
were wrong, each corrected by the user or Team Beta. Every one would have been prevented by
a single grep. This skill encodes the facts that were gotten wrong and the procedure that
prevents it.

**Currency:** updated through Phase 6.0 close (2026-07-30, HEAD ≈ `fbac058`). Anything
after that date is unverified by this document.

---

## 1. THE RULE (non-negotiable)

**Never assert that anything in this project is missing, broken, unwired, unused, current,
or superseded without a `file:line` anchor obtained in THIS session.**

Trigger phrases that require verification before the sentence is written:
"there is no…", "nothing does…", "X is not wired", "X is missing", "the objective is…",
"the target is…", "X was never built", "X already exists", "we should add X".

If verification isn't possible (no repo access, tool unavailable), the claim must be
labeled inline: **[UNVERIFIED]**. Do not launder a guess as a finding.

**Corollary — trace the whole path.** Never classify a capability from one module. Follow
producer → artifact → consumer. Real code with no producer for its input is *not* "wired."

**Corollary — check for supersession, not just existence.** This project is 180+ sessions
old. Finding something in the repo proves nothing about whether it is live. Before citing
any metric, target, doc, or script as current, grep for its replacement.

**Corollary — the repository is NOT the system (VIR-6).** Provisioning state — systemd
units, cron, host config, deployed-but-uncommitted files — is invisible to every gate in
this project, all of which read git. A repo-scoped search may never be reported as a
system-scoped result. This corollary exists because Alpha reported "no scraper invoker
exists" after searching a clone; an enabled boot-triggered systemd unit was found on the
host minutes later.

**Corollary — read the consumer before specifying a change.** Before changing any shared
buffer, on-disk path, or data format: enumerate every producer, consumer, owner, lifecycle
transition, path-type requirement, and downstream schema dependency. For path changes also
check regular-file vs symlink ownership, compatibility aliases, atomic-replace boundaries,
cleanup behavior, concurrent-run namespace, and restart consumers. Three briefs in a row
were defective for skipping this.

---

## 2. SETTLED FACTS (verified; do not re-derive)

### 2.1 Objective / target lineage — TWO pivots
```
score (training match rate)   → superseded
holdout_hits (holdout Hit@K)  → superseded as ML target
holdout_quality (composite)   → replaced holdout_hits
```
- Chapter 6: y-target moved from `score` to `holdout_hits`.
- Chapter 4: `holdout_hits` was "the ONLY non-circular training target."
- **Chapter 14:** *"holdout_hits as ML target produced R2=0.000155 on real CA Daily 3 data —
  zero signal. holdout_quality (composite score) replaced holdout_hits."*
- **R² IS NOT THE PROJECT OBJECTIVE.** Abandoned for measuring zero signal.
- Pipeline objective (S140b), implemented in `chapter_13_orchestrator.py` and
  `database_system.write_downstream_score`:
  ```
  downstream_score = 0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·pool_coverage
  ```
- `evaluate_pools.py` already computes coverage + **lift vs random**.
  **Do not propose building a coverage metric — it exists.**
- **Known open discrepancy:** selfplay inner-loop fitness still uses R²;
  `selfplay_orchestrator.py:933` still prefers `holdout_hits`. A finding, not intended design.

### 2.2 Feature contract
- **91 features extracted / 89 trained** (excluded: `score`, `confidence`).
- "~62 features" is **STALE** — `feature_importance.py:95-119` omits 31 live features.
- 5 dead placeholders with **no producer**: `skip_mean`, `skip_std`, `skip_entropy`,
  `survivor_velocity`, `velocity_acceleration`.
- 14 `global_*` are stamped identically on every survivor in a run → no within-run
  discriminating signal; random row-level folds across runs can leak run identity.

### 2.3 The 22-array NPZ contract (`[S172 Phase-5 D3.0]`, frozen)
- Only **4 columns carry per-seed information**: `seeds`, `forward_matches`,
  `reverse_matches`, `score`. 10 are trial-level aggregates, 6 run/config constants,
  2 categorical labels. `intersection_count` duplicates `bidirectional_count` (deliberate).
- **`forward_matches`/`reverse_matches` are the only independent per-seed sieve signal and
  are NOT in the Step-3 merge list.** Producer/consumer semantic divergence under one name.
  Introduced `0e82155`.
- **RANGE-MINER obligation: emit exactly the 22 arrays, nothing more.**

### 2.4 The bidirectional sieve — reverse is BY DESIGN
- `*_reverse_sieve` kernels iterate the PRNG **forward**; direction comes from
  `residues[::-1]` on the host. There is **no inverse LCG** — intentional.
- **DO NOT flag this as a defect.**

### 2.5 Attribution status
- `per_survivor_attribution.py` implements genuine single-survivor attribution for all four
  model families; Chapter 13 calls it with seed identity.
- Correct status: **implemented, invoked, unreachable, unconsumed.** Four blockers (no
  producer for `predictions/ranked_predictions.json`; `chapter_13_orchestrator.py:582` reads
  `feature_names` at the wrong nesting level; Ch13 bypasses the canonical NN loader; NN
  attribution omits the training scaler).
- **Never call it "wired" or "not implemented."** Both are wrong.

### 2.6 Selfplay
- A **policy-conditioned evaluation harness**, not a learning system. `propose_transform_update`
  is a no-op placeholder. **Do not describe selfplay as "the TFM learning system."**
- Promotion seam is **broken**: `chapter_13_acceptance.py:224` `SelfplayCandidate` lacks
  `transforms`/`fingerprint`/`parent_policy_id`.
- All Ch13 retrain triggers are **defensive**; no opportunity-seeking trigger exists.

### 2.7 Recurring defect pattern — "configured parameters don't reach kernels"
Appears at least four times independently. **One is fixed; three remain open.**

| instance | status |
|---|---|
| RANGE-MINER miner filtered at hardcoded `0.25` | **FIXED** — D6 correction, committed `2be51d5`. Single canonical path, per-direction resolution in the parent, effective value read back off the executor, parent-side fail-closed provenance enforcement. |
| Optuna thresholds never reach the sieve — `test_config` default args `ft`/`rt` never supplied, so **every trial runs at 0.30/0.30** while the study records suggested values | **OPEN [reverify]** — this is the *window-optimizer/PWC* path, not the miner. **Bears directly on Phase 6:** if PWC runs at 0.30/0.30 while the miner honours configured thresholds, the four-path comparison will diverge and look like a miner defect. Verify before Phase 6. |
| Hybrid kernels hardcode `expected_skip = 5`; `skip_min`/`skip_max` are not kernel parameters | **OPEN [reverify]** |
| Variable-skip passes run at hardcoded `0.50` while constant passes run `0.30` | **OPEN [reverify]** |

Fix pattern (Team Beta, D6): **one canonical path** — resolve once in the parent, never
reinterpret downstream, record requested/payload/effective for provenance.

### 2.8 RANGE-MINER Phase 5 — as-built (S172)
All committed and dual-pushed.

| deliverable | what it guarantees |
|---|---|
| D3.5 | shared run finalizer; immutable chain-authenticated generations; owns the root compatibility **symlinks** `bidirectional_survivors_all.npz` / `_binary.npz` and **fails closed if a regular file appears there** |
| D4 | `serial_reference` backend behind a frozen two-backend interface that fails closed |
| D5 | `process_sharded` backend — parallelizes **only** spool-local validation; parent alone owns merge, duplicate attribution, intersection, enrichment. **Available but UNPROMOTED** (~1.6× faster high-survivor at ~2–3× RAM; ~180× slower low-survivor). `serial_reference` remains default. |
| D6 | production integration adapter; miner candidates reach the Step-1 accumulator and a certified generation; directional thresholds reach the kernel with requested/payload/effective provenance and parent-side fail-closed enforcement; shared session-filtered residue authority |
| D6.1 | incremental NPZ checkpoint **writes for the first time** — relocated to run-isolated `.s172_checkpoint/<run_id>/`, open-handle write + fsync, six-field transaction identity, three-tier visible failures. **NON-AUTHORITATIVE four-field snapshot.** |
| Phase 6.0 | RX 6600 XT (ROCm) and RTX 3080 Ti (CUDA) produce a **byte-identical certified artifact** — same `artifact_sha256` across the D6 release-grade generation and both Phase 6.0 runs; 22/22 arrays equal; no GPU reset, no `GCVM_L2_PROTECTION_FAULT`, no VM fault in the host kernel log |

**Authoritative artifact:** D6 release-grade generation
`gen-20260730T002104136270Z-step1_java_lcg_0`, commit `b08c2c5`, `artifact_sha256`
`0e0092fe…c4b0`. Now to be described as the **pre-dataset-provenance authoritative
generation**; its sidecar will not be rewritten. The Phase 6.0 ROCm generation is
**platform-validation, non-authoritative**.

### 2.9 Known-disabled / deliberately off
- **S166 in-memory clear** — disabled. The candidate list is **not** cleared after a flush,
  so RAM growth is unbounded over a long run. Cannot be enabled until the checkpoint carries
  all 24 `CANONICAL_RECORD_FIELDS` (the snapshot carries 4). **D6.2 blocker for Phase 7.**
- **`.s172_checkpoint/<run_id>/` is never pruned** — unbounded directory growth across a
  soak. **D6.3 blocker for Phase 7.**
- **`process_sharded`** — selectable, unpromoted.
- **`/etc/systemd/system/daily3scraper.service`** — enabled since Sep 2025 with
  `Restart=always`, target `run_daily3scraper.py` **never existed** (not in git history);
  produced an ENOENT restart loop every boot. **Now `disable --now`; unit file retained**
  as intentional pre-wired infrastructure. Must stay disabled until Phase 6-P2 is certified.

### 2.10 Dataset lifecycle (Team Beta rulings, 2026-07-30)
- The draw dataset is **mutable**. The old scraper had append **and rewrite** modes; rewrite
  was a one-time bootstrap workaround. **The rewrite mode is being eliminated** — the new
  scraper will publish **immutable versioned files** with an **atomic pointer-manifest file**
  (not a bare symlink; a symlink may exist for operators but must not be a second authority).
- **Version IDs need UTC timestamp + content identity** — "dated" alone is insufficient
  (multiple publications can occur on one date): `daily3-20260730T184233Z-<sha256-prefix>.json`.
- **Two separate walls, do not conflate:**

| wall | proves | passes when |
|---|---|---|
| publication prefix | history was not rewritten | old **canonical record sequence** is an exact prefix of new |
| accumulator input | rows share one computational input meaning | exact input-manifest digest match |

- **A byte-prefix test is invalid for JSON arrays** — a complete array ends with `]`, so
  yesterday's document is never a byte prefix of today's. Use a parsed record-sequence check.
- **Append-only does NOT make prior scores valid on the next version.** Adding a draw changes
  the active temporal window, survivor eligibility, gap/skip features, global frequency and
  residue features, normalization, and any "latest N draws" calculation. **Prefix-only
  accumulator merging is NOT approved.** Different input digest → clean accumulator lineage
  or governed re-evaluation of every retained row.
- **Historical corrections** publish a corrected complete snapshot as a **new dataset
  lineage**; the old lineage is preserved and audit-retained. The scraper must stop with
  `CORRECTION_REQUIRED` and may **not** create the corrected lineage autonomously.
- Generations **chain** — the finalizer merges current winners with the prior generation's
  rows (`prior_rows`). This is why input identity is a lineage invariant, not annotation.

---

## 3. SUPERSEDED — present in repo, NOT current

| Artifact | Status |
|---|---|
| R² as objective | Abandoned (zero signal on real data) |
| `holdout_hits` as ML target | Superseded by `holdout_quality` |
| `feature_importance.py` 60-name list | Stale by 31 features |
| "~62 features" | Wrong; 91/89 |
| `bidirectional_survivors.json` | No longer survivor data; 6 consumers still treat it as survivors |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | Fragment only |
| `run_full_scoring.sh` | Superseded by `run_step3_full_scoring.sh` v2.0.0 |
| PWC / ZMQ transports | Superseded by RANGE-MINER (kept flag-selectable as Phase-6 oracles) |
| `full_scoring_worker.py` docstring "50 features" | Stale |
| `window_optimizer.py:450` docstring | Contradicts live behavior |
| `RUNTIME_DATASET_PROVISIONING_CONTRACT.md` `expected_sha256` as static config | **Superseded** — assumes an immutable dataset; replaced by run-scoped frozen identity resolved from the pointer manifest |
| scraper `--rewrite` mode | Being eliminated; must not reappear under another name |
| "RX 6600" on rrig6600 | The cards are **RX 6600 XT** (32 CUs, VBIOS `…XT…`, verified via `rocm-smi --showhw`). Do **not** globally rewrite other nodes — inventory each from KFD/sysfs. |
| "the writer is unconditionally frozen" (post-D5) | D6 added one approved extension: an optional assembly-backend seam whose `None` path is the exact pre-D6 behavior |

---

## 4. FROZEN — reuse, never reimplement or modify

- **`_l2_sort_key` / `_select_l2_winners`** (`utils/run_finalizer.py:690`, `:714`) — Ruling D.
  Canonical one-record-per-seed rule: highest **float32** score → lowest `trial_number` →
  constant-before-variable *within a trial only*. Same-trial/same-mode collision raises
  `AccumulatorConsistencyError`. Comparing pre-rounding float64 **is the defect this converts
  away**. Any reconciliation must call this authority; if importing inverts a dependency,
  extract to a shared module — never fork.
- **D3.5 finalizer-owned root symlinks** — `bidirectional_survivors_all.npz` /
  `_binary.npz`. A regular file appearing there makes `finalize_run` raise `PublicationError`.
  Nothing else may write those paths.
- **D5 §6.7.A compressed-artifact ban** — scoped to worker transport artifacts
  (`miner/assembly_shard_worker.py`), enforced by mutant M6a on `compress_type=8`. The D6.1
  **checkpoint may be compressed**; the two contracts are deliberately separate. Do not
  "harmonize" them.
- **The 22-array NPZ contract** and the D3.25 four-map ingress contract.
- **`distributed_config.json` bare-metal addresses** (`.120/.154/.162`) — deliberate; they
  match the *default boot target*. `CLAUDE.md` §3 states explicitly this is **not a bug and
  must not be corrected.**

---

## 5. VERIFICATION INTEGRITY (VIR-1…6) — binding

Full text: `docs/VERIFICATION_INTEGRITY_STANDARD.md`. Adopted after three incidents of *a
check that was not checking, presenting as a check that passed* (a vacuous mutant; a flush
that failed for months behind a non-fatal warning; a cleanup `pkill -f` that killed its own
reporting shell).

- **VIR-1** Verification must prove its own execution. Silence, truncation, reporter death,
  or an inaccessible surface may never be read as a pass.
- **VIR-2** Potentially vacuous detectors need: execution proof · **clean control** (does not
  fire when clean) · **fault-injection/positive control** (does fire on an injected defect) ·
  detector independence. *"Positive control" = fault injection; "negative/clean control" =
  no-defect. Not interchangeable.*
- **VIR-3** Terminate in `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. Only `PASS` satisfies an
  acceptance item. A missing completion sentinel is failure, never success.
- **VIR-4** Cleanup must not be able to kill its reporter (no `pkill -f` from a shell whose
  own command line matches the pattern).
- **VIR-5** Unobservable is not clean. An empty container kernel log when host logs are
  inaccessible is `UNAVAILABLE`, not clean. RAS counters absent on unsupported hardware are
  `UNAVAILABLE`, not "zero errors."
- **VIR-6** Audit scope must match the claim. Declare searched surfaces, unavailable
  surfaces, method and bounds. A repo-scoped audit may not be reported as system-scoped.

Every brief carries:
```
Verification-integrity controls (VIR-1…6):
- execution proof:        - clean control:        - fault-injection control:
- completion sentinel:    - unavailable-observer behavior:
- audit claim scope:      - searched surfaces:    - unavailable surfaces:
```

---

## 6. TOPOLOGY (verified 2026-07-30)

Rigs boot **bare-metal by default**; they are currently booted into **Proxmox**.
Pattern: `host = rig + 1`, `CT100 worker = host + 1`.

| rig | bare-metal | Proxmox host | **CT100 worker (use this)** |
|---|---|---|---|
| rrig6600 | `.120` | `.121` | **`192.168.3.122`** |
| rrig6600b | `.154` | `.155` | **`192.168.3.156`** |
| rrig6600c | `.162` | `.163` | **`192.168.3.164`** |

- All three CTs: key auth from **VM101** works (`BatchMode=yes`), `~/rocm_env` present,
  **8 × RX 6600 XT** visible, cupy 13.5.1 under ROCm, gfx1032, **no HSA/GFX env overrides
  needed or set** (ROCm 6.4 supports gfx1032 natively).
- **Venvs differ:** VM101 uses `~/venvs/torch`; the rigs use `~/rocm_env`.
- CT100 is an **unprivileged LXC** — `dmesg`/`journalctl -k` unavailable inside; GPU kernel
  log must be read from the **Proxmox host** (`root@.121`), where `amdgpu` lives.
- `daily3.json` is **gitignored** — `git clone` alone cannot stand up a rig.
- Key auth works **from VM101**, not from ser8. Commands must state their host.

---

## 7. WORKING AGREEMENTS

- **Claude = Team Alpha** (lead dev/implementation). **Team Beta** = separate approval
  authority; rulings binding. Never impersonate Team Beta.
- **Never commit or push from the sandbox.** Deliver to `/mnt/user-data/outputs/`; Michael
  downloads to ser8, `scp`s to VM101, commits, dual-pushes.
- **Every command must name its host.** ser8 = download target and `scp` source. VM101 =
  the repo, all `git`, all rig SSH. Rigs = workers only.
- **VM101 (`192.168.3.177`) is the live dev box.** Bare Zeus (`.127`, `rzeus`) is the
  FROZEN FALLBACK — never scp working files there.
- Use **absolute paths** for scp to VM101.
- **Stage explicitly, never `git add -a`** — the tree usually has in-flight work.
- Write a `SESSION_CHANGELOG_YYYYMMDD_SN.md` each session; commit to `docs/`; dual-push
  (`git push origin main && git push public main`). **Both remotes are public-facing in
  effect** — `origin` is private, `public` is a public mirror; everything pushed goes to both.
- Prefer **Claude Code on VM101** for any as-built question — it reads live source **and the
  live host**. A public clone is repo-only (VIR-6). Chat-side reasoning is provisional.
- Every Claude Code brief: **one falsifiable question, a defined deliverable, and "write the
  report to `docs/<n>.md`"**. Separate *investigate* briefs from *fix* briefs.
- Long GPU/nested suites (D5 drags the whole chain) look hung under `| tail`. Check for a
  **descendant** process before concluding a hang; a blocked parent burns no CPU.

---

## 8. APPROVED SEQUENCE (Team Beta)

```
D6.1 ✅ · Phase 6.0 ✅
Phase 6-P0  freeze publication/pointer/correction schemas; bootstrap publication;
            pointer resolution, fleet provisioning, fail-before-dispatch
Phase 6-P1  dataset provenance binding + exact-input accumulator wall
bounded multi-rig Phase 6   (four-path verify; §17 promotion benchmark)
D6.2 · D6.3 · Phase 6-P2    (relative order flexible)
Phase 7     26-GPU saturation + WATCHER soak
```
**Hard Phase 7 prerequisites:** 6-P0, 6-P1, D6.2, D6.3, 6-P2.
**Bounded Phase 6 conditions:** fresh process or reset accumulator per scenario; declared max
seed/survivor volume; host-RSS monitoring; no reliance on list clearing; no resume/restart
acceptance claim; no WATCHER long-lived loop; no multi-trial soak. If Phase 6 becomes
long-lived with unbounded list growth, **D6.2 moves in front of it.**

---

## 9. SELF-CHECK BEFORE SENDING

1. Did I claim anything is missing/broken/unwired/current? → anchor or mark **[UNVERIFIED]**.
2. Did I cite a metric or target? → is it on the supersession list (§3)?
3. Am I proposing something that already exists? (coverage metric, downstream_score,
   attribution engine — all already built).
4. Did I classify a capability from one module without tracing producer → artifact → consumer?
5. Am I about to change a shared buffer, path, or format? → did I enumerate every consumer?
6. Is my claim system-scoped but my evidence repo-scoped? (VIR-6)
7. Did I name the host for every command?
8. Am I inflating a doc/code divergence into an architectural defect before checking the
   design rationale docs?
9. Is this a long thread? Long context degrades verification discipline. Say so and suggest a
   fresh session for major new work.
