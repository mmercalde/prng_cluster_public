# SKIP — the design record, for the owner's RANGE-MINER skip-range decision (F-1)

**Date:** 2026-11-21 · **Host:** VM101, as `michael` · **Read from:** the main tree
`/home/michael/distributed_prng_analysis` at `6d75f71`, read-only · **Written in:** the worktree
`/home/michael/tfm_worktrees/pwc-offset` (this one file only). **Type:** read-only research. No
code was changed, nothing was committed, nothing was relayed to Beta.

**Asked by the owner, 2026-11-21 08:50 PST:** "TFM is such a large project with so many revisions
and pivots I dont remember how we envisioned it and what it is supposed to do- can you read the
repo docs- maybe confer with beta?"

**Beta was not consulted.** A relay needs the owner's approval. §5 quotes every recorded Beta
position on this question verbatim. A relay with this file attached is one of the options in the
summary.

---

## Summary for the owner (one page)

**What skip is for.** The published Daily 3 numbers are not every output of the drawing machine.
Before each automatic draw the machine runs an unpublished pre-test. The evening session also
draws Daily 4, Fantasy 5 and Daily Derby on the same machine. So between two Daily 3 values you
can see, an unknown number of machine outputs were used and never published. **Skip is the
sieve's model of those gaps.** Two modes test two shapes of gap:
- **Constant skip, phases 1–2:** the same gap every time. The kernel tries every gap from
  `skip_min` to `skip_max` for each seed and keeps the best (§2).
- **Variable skip, phases 3–4:** the gap changes from draw to draw. The kernel searches near a
  running estimate that starts at 5.

**What the design says phases 1–2 should do with the trial's skip range.** Every design source
found says the constant kernels search the range the trial was given:
- the input docs: `skip_min`/`skip_max` are "Minimum/Maximum skip value in pattern", with default
  0 and 16;
- the Optuna search space, which samples `skip_min` ∈ [0, 10] and `skip_max` ∈ [10, 250];
- Chapter 1 ("Constant-skip is fully wired");
- Chapter 2 ("The bounds reach the kernel");
- the 2026-07-31 audit (row M6a: "reaches the kernel");
- the Phase-6 submission ("Sampled `skip_min`/`skip_max` are placed in the payload's `skip_range`").

The two engines that came before RANGE-MINER, the legacy coordinator and PWC, both send the
trial's range. **No document found says `[0, 16]` on phases 1–2 is intended.**

**What RANGE-MINER does today.**
- **The gap:** the coordinator's stripe payload has never carried a `skip_range`. A pickaxe over
  every commit of `miner/range_miner_coordinator.py` finds none. So the worker uses its fallback,
  `[0, 16]` (`miner/range_miner_worker.py:1293`).
- **What actually runs:** phases 1–2 always search skip 0–16, whatever Optuna sampled. Run 4's
  pinned 6–99 is recorded in the metadata, and the stamps reach the ML features. The constant
  kernels never search 6–99.
- **Phases 3–4:** they ignore the trial's range on every engine. That is a separate, known, open
  defect.
- **Where `[0, 16]` comes from:** it is the documented default, copied as a fallback.
- **When it was found:** test code knew on 2026-08-01, and Claude Code recorded it as F-1 on
  2026-11-14.

**What is unknown.**
- **Hit rate:** whether delivering the trial's range on phases 1–2 changes pool hit rate. Never
  measured.
- **Survivor counts:** how many phase 1–2 survivors a wider range produces. At Run 4's geometry
  they are 0 today, measured at `[0, 16]`.
- **Pool meaning:** what the past miner-route Optuna studies mean. Their recorded skip values,
  e.g. attempt 3's [7, 229], never governed the constant phases. For phases 1–2 those two
  dimensions were dead on that route.
- **Step 6:** whether a per-seed skip survives to prediction generation on the miner route
  ([UNVERIFIED]; §4.4).

**What would settle it.** Run the same job twice: once as today (`[0, 16]` on phases 1–2) and
once with the trial's range delivered, as a flag-selectable configuration. Everything else stays
pinned. Carry both to prediction pools. Compare each pool's hit rate against `k/1000` on draws the
sieve never saw. Beta recorded the same measurement twice (§5).
- **Not a test:** whether any "true seed" is found or survives. That is the yardstick the project
  rejected.
- **Already known, no test needed:** the cost, at Run 4's geometry (`k=12`, `generator_phase=0`),
  corrected per Beta X-SKIP-F1 R1 Finding 2 (`:50`, `:60`):
  - 6–99 tries **5.53×** the skip hypotheses per seed (94 against 17);
  - it needs **33.1×** the Java LCG source-level state advances per seed (65,283 against 1,972),
    because each hypothesis costs `12 + 13 × skip` advances.

  Fleet timing is **UNAVAILABLE**. These are arithmetic counts, not measurements.

**Decisions that are yours** (also in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`):
- **How to fix:** direct fix, or flag-selectable with `[0, 16]` kept as the baseline (CLAUDE.md
  yardstick step 4).
- **When, operationally:** Run 4 is not still to come (§4.5). Its upper-half attempt 3 earned
  credit on 2026-11-15. Attempt 9 covered the lower half, and later runs under the Run 4 pins were
  clean on both halves. All of them ran phases 1–2 at `[0, 16]`. So the question is where the fix
  sits in the operational sequence: for example, after PWC Stage 4, whose engine comparison pins
  `[0, 16]`. The fix would then be judged against these runs as the historical operational
  baseline.
- **Beta:** whether to relay this record to Beta for a position.

---

## Prior record

**Searches run** (main tree at `6d75f71`; `/bin/grep`, which does not honour `.gitignore`, so
ignored `*.json` and `.bak` files are included):

```
/bin/grep -n -i skip  docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md docs/PIPELINE_BEHAVIOUR_MODEL.md
                      docs/CHAPTER_1_WINDOW_OPTIMIZER.md docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md
                      docs/HYBRID_SKIP_BOUND_AUDIT.md docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md _v1_4_5.md
                      docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md
                      config_manifests/parameter_registry.json config_manifests/feature_registry.json
                      agent_manifests/window_optimizer.json distributed_config.json
/bin/grep -l -i skip  docs/TB_RULING_*.md      (then read the skip lines of each hit)
/bin/grep -rln -e '\[0, *16\]' -e '0–16' -e '0-16' docs/        (40 files; the skip-relevant hits read)
/bin/grep -n -i "skip_range|skip range" docs/LEADS.md docs/TODO_CONSOLIDATED.md
             docs/exchange/from_beta/X-VOLUME-LIMIT_R1/R2_BETA_REVIEW.md X-PWC-STAGE3_R1_BETA_REVIEW.md
/bin/grep -rln '"skip_range"' scripts tests utils ; /bin/grep -n skip_range miner/*.py coordinator.py
             persistent_worker_coordinator.py persistent/pwc_worker_service.py sieve_gpu_worker.py
git log --all -S'skip_range' -- miner/range_miner_coordinator.py    → no commit, ever
git log --all -S'skip_range' -- 'miner/*'                           → dbe3d0e, b9c6120, 70cd6f0
git blame -L1293,1293 miner/range_miner_worker.py                   → dbe3d0e, 2026-07-18
git log -S"which carries no skip_range" -- tests/phase6/known_answer_gate.py → d98298c, 2026-08-01
pdftotext docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf  (read §II, §V pre-test, §XI note)
/bin/grep -rln -i "skip_range|skip range" ~/dashboard_work   (host-state brief corpus)
```

**Documents that already decide or record this subject:**

| Document | What it records | Label |
|---|---|---|
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (item 7) | F-1: phases 1–2 never receive the skip range; "This is the owner's decision, after Run 4." | ALREADY RECORDED, open |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:360-370`, `:389-405` | F-1 in full, with the measurement that would settle it | ALREADY RECORDED |
| Beta `X-VOLUME-LIMIT_R1:79`, `_R2:80`; `X-PWC-STAGE3_R1:115` | Beta's status statements (§5) | ALREADY RECORDED |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` (16:45 and 21:32-21:35 entries) | Owner: PWC keeps the trial's range; "range miner must be fixed none the less" | Owner rulings |
| `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md` | The fix's scope and the owner questions | ALREADY RECORDED (landed `390c285`) |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md`; Chapter 1 §3.1.1 (D-1/D-2); Chapter 2 §5.4 | Phases 3–4 ignore the skip range on every engine | ALREADY RECORDED, open, separate |
| Chapter 2 §5.1/§5.6; `PIPELINE_BEHAVIOUR_MODEL.md` §11 | Why skip exists; "the fix is WIRE-IN, not removal" | ALREADY RECORDED |

**Labels for what this record raises:**
- F-1 itself: ALREADY RECORDED.
- **NEW to the written record:**
  - how the documents came to contradict the code (§3.3): a test harness added `skip_range`
    itself, while a sibling gate in the same commit noted its absence;
  - that the miner-route Optuna skip dimensions were dead on phases 1–2 as well as 3–4 (§4.5);
  - the `uint8` question, resolved (§4.2).
- Nothing here CHANGES A RULING.

**Not searched:**
- **Places:** ser8 archives, the private repository's history beyond what this clone holds,
  claude.ai project files, the rigs, the Proxmox hosts.
- **Search limits:**
  - `docs/LEADS.md` was searched by keyword only. No skip-range entry was found.
  - Of the 40 files that mention `[0,16]`, only the skip-relevant hits were read.
  - `docs/Cluster_operating_manual.txt` was searched for `[0,16]` only.

---

## 1. Why skip exists at all

**The physical model, from the source document** (`docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf`,
committed `746b545`; earlier chapters marked it UNAVAILABLE because it was not yet in the repo):

- **The pre-test, §V steps 14-16 (PDF p.5-6):**
  - "click the green [Start Draw Session] button to run Pre-Test";
  - "After the pre-test draw(s) is complete, remove the check mark (✓) next to 'Run Draw as Test'
    to run an official draw";
  - "Document pre-test details if an additional pre-test(s) are conducted for any reason; a Draw
    Anomaly report is required."
- **The game set, §V step 12:** one session's draw specs list "CA Fantasy 5 05:01-39u · CA Daily
  4 04:00-09r · CA Daily 3 03:00-09r · CA Daily Derby 03:01-12u 03:00-09r".
- **Equipment, §II:** "A random number generation (RNG) program is used to select the primary and
  alternate draw equipment which will be used for the draw."
- **Two generators, §V step 4:** "Verify both A and B RNG icons displayed are green."

**How the repository states it:**
- **The model:** "The published draw sequence is not an uninterrupted PRNG output stream … Skip
  models those gaps. It is a physical property of the data source, not a tuning convenience."
  (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §5.1; same text at `docs/PIPELINE_BEHAVIOUR_MODEL.md`
  §11.1, `:794-822`.)
- **Its stated limit:** "These are therefore physically motivated *candidate gaps* supporting skip
  as a detector — not proven state advances." (Chapter 2 §5.1.)
- **The standing rule:** "**The correct action on finding skip bounds unwired is to wire them in,
  not to remove them**" (Chapter 2 §5.1.1). Also: "**the fix is WIRE-IN, not removal.**"
  (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:270`; `PIPELINE_BEHAVIOUR_MODEL.md` §11.2.)
- **Intent:** "The goal was never to reverse state. It is to extract a fingerprint." Chapter 2
  §5.6 records this as the owner's design doctrine, accepted by Beta, and says the framing itself
  is "NOT FOUND anywhere in the repository" before that section.
- **The whitepaper** (`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`): **NOT FOUND.** It
  contains no occurrence of "skip". Its role is the bidirectional and loose-threshold argument,
  not the gap model.

## 2. What each phase is meant to do with skip, and what Step 1 tunes

### 2.1 The input reading, the one F-1 is about

- **`docs/instructions.txt:1181-1183`** (Oct-2025 operator spec; also in an older revision, per
  `docs/SKIP_SEMANTICS_SEARCH_v1.md:40-42`):
  - "`--offset INT`: Temporal alignment (PRNG steps to skip before sequence)"
  - "`--skip-min INT`: Minimum skip value in pattern (default: 0)"
  - "`--skip-max INT`: Maximum skip value in pattern (default: 16)"

  This is where `[0, 16]` comes from: the documented default.
- **`config_manifests/parameter_registry.json:156-167`:** `skip_min` default 0, "Minimum skip
  value for sieve search"; `skip_max` default **20**, "Maximum skip value for sieve search".
- **`config_manifests/feature_registry.json:329-354`:** the output reading. "Minimum/Maximum skip
  value found during sieve analysis (from Step 2)". The two registries disagree; this is recorded
  as DIVERGENT D12 (`PIPELINE_BEHAVIOUR_MODEL.md:1203`).

### 2.2 Constant skip, phases 1–2

- **The kernel:** it loops `for (int skip = skip_min; skip <= skip_max; skip++)` and keeps the best
  skip per seed (`prng_registry.py:960-995`; loop at `:972`). It writes `best_skips[pos] =
  (unsigned char)best_skip_val` (`:461`).
- **Chapter 2 §5.2:** "22 of 22 constant kernels declare `int skip_min, int skip_max` and search the
  whole range per seed … The bounds reach the kernel: `_constant_prefix` emits them". The first
  half is true at the kernel ABI. The second half is contradicted on the miner route (§3.2).
- **Chapter 1:** "Constant-skip is fully wired on all four `java_lcg` variants; the variable-skip
  path is where the loss is." (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:315`.) Same contradiction.

### 2.3 Variable skip, phases 3–4

- **What the kernel does:** "No pattern is supplied and none is generated. `skip_sequences` is an
  output … a greedy per-draw adaptive search … `expected_skip` is hardcoded to 5" (Chapter 2 §5.3;
  `prng_registry.py:1027`).
- **The open defect:** 0 of 22 hybrid kernels take `skip_min`/`skip_max` (Chapter 2 §5.4). Beta's
  callout, verbatim: "DEFECT — current hybrid kernels do not execute the requested
  skip_min/skip_max semantics and instead use a hard-coded stride. Hybrid optimization results are
  non-certifying." (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:255-261`.)
- **Separate from F-1:** it applies on every engine. Beta ruled it out of the window-anchor scope:
  "`(c)` `skip_min`/`skip_max` hybrid-search semantics stays outside this proposal and remains on
  its existing sampler-comparison chain" (`docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:37`).

### 2.4 What the Step 1 optimizer is meant to tune

- **The dimensions:** seven sampled dimensions, including `skip_min` and `skip_max`
  (`PIPELINE_BEHAVIOUR_MODEL.md` §3.2, `:242-251`): "`skip_max`'s floor is `max(skip_min,
  bounds.min_skip_max)`, so the two skip dimensions are coupled."
- **The live bounds** (read by `window_optimizer.py:166-170` → `SearchBounds`):
  `distributed_config.json:84-91` gives `"skip_min": {"min": 0, "max": 10}` and
  `"skip_max": {"min": 10, "max": 250}`.
- **The code fallback** is the same: `window_optimizer.py:75-76`, `:144-147`, commented "S139:
  500→250".
- **Sampling:** `window_optimizer_bayesian.py:549-554` calls `trial.suggest_int('skip_min',
  bounds.min_skip_min, bounds.max_skip_min)` and `trial.suggest_int('skip_max', max(skip_min,
  bounds.min_skip_max), bounds.max_skip_max)`.
- **Not a live bound:** `agent_manifests/window_optimizer.json:98-115` documents `skip_min` 0–50
  and `skip_max` 20–500 (default 100) under `parameter_bounds`. Neither key is in that manifest's
  `default_params` or `args_map` (checked by loading the JSON). The only Python readers of
  `parameter_bounds` found are tests, `parameter_advisor.py:1150` and the miner coordinator's
  precedence note (`:12470`).
- **The bounds' origin:** no derivation found for the value 250 or the 10/10 floors, beyond the
  "S139: 500→250" comment. The project-facts skill §2.20 records the same: "No bound's VALUE has a
  derivation anywhere".

### 2.5 The Phase-6 and audit statements the code contradicts

- **`docs/HYBRID_SKIP_BOUND_AUDIT.md` §2.4** (2026-07-31, HEAD `8a55a68`):
  - M4 says "`miner/range_miner_worker.py:776` — `skip_min, skip_max =
    tuple(payload.get("skip_range", [0, 16]))` | worker unpack | carried";
  - M6a says "`ScalarArg(ctx.skip_min,"int32")`, `ScalarArg(ctx.skip_max,"int32")` … | **CONSTANT**
    ABI | ✅ **reaches the kernel**".

  The table goes from M3 (coordinator and manifest) straight to M4 (worker unpack). It has no
  payload-builder hop.
- **`docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382`:** "effective skip semantics | **constant
  only.** Sampled `skip_min`/`skip_max` are placed in the payload's `skip_range` and reach `for (int
  skip = skip_min; skip <= skip_max; skip++)`." Why this was true of the harness and not of
  production is §3.3.
- **TB_RULINGs:** no ruling decides the constant-phase skip range. The Run-4 ruling approves
  "`skip_min = 6` · `skip_max = 99`" as part of the pinned geometry
  (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:16`, `:141-142`). The window-anchor rulings
  hold "skip geometry … constant between governed and control arms"
  (`docs/TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md:17`, `:60`).

## 3. History

### 3.1 What the engines before RANGE-MINER did, and what PWC does now

All three non-miner routes send the trial's range on the constant phases, with `[0, 16]` only as a
worker-side fallback when the key is missing.

- **Legacy coordinator:** `coordinator.py:1480` has `'skip_range': [args.skip_min, args.skip_max]`.
  It also builds the range from `args` at `:1383-1384` and `:2331`, with `[0, 20]` sieve defaults
  (`:346`; `distributed_config.json:51-54`).
- **Legacy worker:** `sieve_gpu_worker.py:155` has `skip_range = tuple(job.get('skip_range', [0,
  16]))`, used at `:193`. This line is in the file's first commit in this repository, `f300396`
  (2026-03-09), so its origin predates the repo copy.
- **PWC:** `persistent_worker_coordinator.py:2639` has `_skip_range = [config.skip_min,
  config.skip_max]`, passed to every pass at `:2718`, `:2745`, `:2790`, `:2815` with the comment
  "[PWC-BACKUP S4] no fallback". `run_sieve_pass` refuses a missing range (`:2046-2051`).
- **PWC worker service:** `persistent/pwc_worker_service.py:552`.

### 3.2 How RANGE-MINER came to use `[0, 16]`

- **2026-07-18, Phase 3, worker `dbe3d0e`** ("per-GPU worker daemon (Team Beta approved
  rev-3)"). `git blame` puts `skip_min, skip_max = tuple(payload.get("skip_range", [0, 16]))`
  (now `miner/range_miner_worker.py:1293`) at this commit. It is the same expression and default
  as `sieve_gpu_worker.py:155`. The spec required the hybrid ABIs to be replicated verbatim
  (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:368`: "Miner must replicate verbatim"). That the
  fallback was copied with them is an **inference** from the identical text; no brief says so.
- **The coordinator payload:** `build_stripe_assign_payload` (`miner/range_miner_coordinator.py:9216`,
  dict at `:9331-9358`) holds dataset, sha256, window size, sessions, `window_anchor`,
  `generator_phase`, the residue sha256, and the two thresholds. It has no skip key.
  `git log --all -S'skip_range' -- miner/range_miner_coordinator.py` finds **no commit ever**.
  `StripeAssignMessage` (`miner/range_miner_protocol.py:97-110`) has no skip field either; its
  `payload` dict is documented as "window cfg, thresholds, residue ref".
- **2026-07-22, D0 `4c697a8`:** the skip bounds were threaded into the miner, **as metadata**.
  - `window_optimizer_integration_final.py:1979-1985`: "D0 seam: the skip bounds were the one
    WindowConfig pair still dropped here … thread them from the resolved WindowConfig so every
    published miner manifest carries the real skip range".
  - `miner/range_miner_coordinator.py:12390-12400`: "skip bounds flow from the resolved
    WindowConfig … so every published manifest carries the real skip_min/skip_max".

  The stated purpose is the published manifest. The payload was not touched.
- **The RANGE-MINER spec** (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`) lists `skip_min`,
  `skip_max` and `skip_range` only as NPZ arrays (`:204-206`, `:622-624`). It says nothing about
  carrying the trial's range to the constant kernels. **NOT FOUND:** a spec line requiring the
  assign payload to carry `skip_range`.

### 3.3 How the documents came to say the opposite (NEW to the written record)

Both of these landed in the same commit, `d98298c` (2026-08-01, "S184 bounded Phase 6").

- **The sampler harness** that produced the Phase-6 submission's TPE-vs-random datapoint calls the
  production builder and then adds the key itself (`tests/phase6/sampler_control_arm.py:160-168`):
  "skip_min/skip_max are LIVE on the constant path — this is the sampled pair reaching the kernel
  … `payload["skip_range"] = [int(cfg.skip_min), int(cfg.skip_max)]`". Its recorded semantics
  string (`:388-393`) is the sentence quoted at `TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:382`.
  The submission's statement was true **of the harness**, not of the production coordinator.
- **The known-answer gate** in the same commit says the opposite, correctly
  (`tests/phase6/known_answer_gate.py:279-284`): "What the WORKER will use — read the way the worker
  reads it, so a verbatim coordinator payload (which carries no skip_range) resolves to the same
  production default [0, 16] on both sides."

So by 2026-08-01 the absence was **known in test code** and called the "production default". It
was **not** carried into any chapter, audit or submission until F-1 (2026-11-14).

### 3.4 Does any document say `[0, 16]` on phases 1–2 is intended?

**NOT FOUND.**
- **Surfaces searched:**
  - the tracked repo: `docs/`, chapters, TB_RULINGs, proposals, exchange, tests, `miner/`;
  - gitignored files: `/bin/grep` reaches them, and the manifests were loaded directly;
  - git history: the pickaxe on the coordinator across all refs, blame on the worker line, and
    deleted files containing `skip_range`, whose names were listed but not each read;
  - the host-state brief corpus, `~/dashboard_work` (3 hits, none on this question).
- **The nearest statements:**
  - the known-answer gate's "production default [0, 16]" (§3.3), which describes rather than
    intends;
  - the documented operator default "(default: 16)" (`instructions.txt:1183`), a default for
    when nothing is sampled;
  - the Run-4 proposal's "**inert for survivor count** — hybrid kernels ignore sampled skip bounds,
    so this is provenance, not a control" (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:61`). It
    reasons about the hybrids only and does not mention phases 1–2.

## 4. What depends on it downstream

### 4.1 The 22-array NPZ and the feature vector

- **The NPZ:** `skip_min`, `skip_max` and `skip_range` are 3 of the 22 frozen arrays, all `int32`
  (`tests/test_s172_phase5_d3_0_encoding_contract.py:97-99`). They are trial-level stamps: every
  row of a trial carries the same values (`utils/canonical_records.py:115-118`).
- **No per-seed best skip:** the NPZ has no per-seed best-skip column. The kernel computes one, and
  the worker collects it (`miner/range_miner_worker.py:1417-1422`, `[int(skip)]` per survivor). No
  NPZ array carries it.
- **Into the ML features:**
  - `full_scoring_worker.py:453-456` copies `skip_min`/`skip_max`/`skip_range` from metadata into
    the feature dict;
  - `survivor_scorer.py:776` merges `skip_min`, `skip_max`, `skip_range` (plus the three dead
    `skip_mean`/`skip_std`/`skip_entropy`).

  The project-facts skill §2.20 places them as features #75/#77/#78 of 89 [UNVERIFIED this session].
- **What that means on the miner route:** those three features describe the **sampled** range,
  not the range phases 1–2 searched. On the hybrid phases this was already recorded as "describes a
  range no kernel saw" (`docs/HYBRID_SKIP_BOUND_AUDIT.md:134`, `:299`). F-1 makes the same true of
  the constant phases.

### 4.2 The `uint8` `best_skips` limit against the search bound (resolves the UNVERIFIED item)

- **The limit:** the kernel stores the best skip as `unsigned char` (`prng_registry.py:960-963`,
  write at `:461`), and the worker allocates `uint8` (`miner/range_miner_worker.py:266`, `:1382`).
  A skip above 255 would be truncated modulo 256.
- **The live bound:** the highest `skip_max` Optuna can sample is **250**
  (`distributed_config.json:88-91`; `window_optimizer.py:76`, `:147`). **250 ≤ 255, so every value
  in today's search space fits.**
- **The remaining exposure:**
  - **The manifest:** its `parameter_bounds` documents `skip_max` up to 500. It is not routed as
    a bound (§2.4), but it would overflow if it ever were.
  - **Operator pins:** a warm-start pin (`--warm-start-skip-max`, `window_optimizer.py:1604`)
    above 255. Whether Optuna's enqueue path accepts an out-of-range pin is [UNVERIFIED].

  Delivering the trial's range on phases 1–2 makes this limit live for the first time on the miner
  route. Today the constant kernels never see more than 16.

### 4.3 The 25,000 survivor cap

- **The decision:** "Owner decision, 2026-11-14: the cap is 25,000 survivors in one phase."
  (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:5`.)
- **What was measured:** Run 4's geometry was measured through the production path, so at `[0, 16]`
  on phases 1–2. Exact counts: phase 1 **0**, phase 2 **0**, phase 3 **60**, phase 4 **23,568**
  (`:13-18`). The estimator says so: "the production assign payload carries no skip_range … so
  they do not reach the sampled kernels — exactly as in a run" (`utils/survivor_volume_estimator.py:661-664`).
- **What a fix changes:** delivering 6–99 changes phases 1–2 only. Phases 3–4, which dominate the
  volume, are unaffected because hybrids ignore the range. The phase 1–2 counts at 6–99 are
  **unmeasured**, so the volume gate must be re-measured after any fix.

### 4.4 Step 6 prediction

- **Code reading:** survivors vote with their own skip. `prediction_generator.py:845-846`: "v6.0:
  Use per-survivor skip for sequence generation … `skip=survivor_skip`". The skip is found under a
  survivor's `skip`, `metadata.skip` or `features.skip_hypothesis` key, else a global fallback
  (`:530-553`, `:109`).
- **[UNVERIFIED]:** whether a per-seed skip reaches Step 6 at all on the miner route. The NPZ has
  no per-seed skip column (§4.1). The Step 3 output was not read this session.
- **Why it matters:** this is "the only place the skip hypothesis does causal work downstream"
  (project-facts skill §2.20).

### 4.5 Optuna and the Run-4 geometry

- **Run 4's geometry** (`docs/TODO_CONSOLIDATED.md:79`): "pinned geometry (k=12, anchor 25, midday,
  0.71/0.47, skip 6–99)". It is an A/B against attempt 9: "attempt 9 … skip 6-99 generator_phase =
  25 / run 4 … skip 6-99 generator_phase = 0"
  (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:140-141`). Both ran phases 1–2 at `[0, 16]`.
- **Run 4 has run** (corrected per Beta X-SKIP-F1 R1 Finding 1, `:59`; citations read for this
  correction):
  - **Upper half, `[2³¹, 2³²)`:** "Run 4 over `[2³¹, 2³²)`: attempt 3 credited clean on
    2026-11-15" (`CLAUDE.md:223`). It is listed as "attempt 3 (17:49, phase 4 17:59:01) … clean,
    credited" (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:154`).
  - **Lower half, `[0, 2³¹)`:** attempt 9 covered it. "the live ledger still contains exactly the
    attempt-9 certified row, `[0, 2147483648)`" (`docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:55`).
  - **Later clean runs under the Run 4 pins:** "both halves of the seed space are covered under the
    Run 4 pins". These are Run L (upper, "CLEAN. 128/128 … phase 3 = 60, phase 4 = 23,568") and
    Run M (lower, "CLEAN. 128/128 … phase 4 = 23,410") (`docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:7`,
    `:14`, `:16`).
- **What that comparison can establish:** the two intervals differ, so the comparison is
  **operational**. "It cannot be described as an experiment with only `generator_phase` changed …
  Record both intervals and narrow the comparison accordingly"
  (`docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:51`). The timing question for the F-1 fix
  is therefore where it sits in the operational sequence, not how it affects an A/B still to run.
- **Optuna studies on the miner route (NEW to the written record):** the recorded best skip ranges
  never governed any phase. Attempt 3's winning trial is recorded as skip [7, 229]
  (`~/dashboard_work/DRAFT_CHAPTER_1_UPDATE.md:25`, host state). On that route the TPE sampler
  associated objective values with `skip_min`/`skip_max` coordinates that phases 1–2 did not
  execute, and phases 3–4 never execute them on any route. This is the same class Beta recorded for
  threshold quantization (skill §2.36) and the dead-dimension rule (skill §0.5). **Not measured:**
  whether it changed which trials won.

## 5. Every prior owner and Beta statement on this question, verbatim

**Owner**
- **2026-11-14, handoff item 7** (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`, written by the
  session): "**The skip range never reaches phases 1 and 2 on the RANGE-MINER route.** The assign
  payload carries no `skip_range`; the worker falls back to `[0, 16]` … Skip is a cornerstone of
  the design. This is the owner's decision, after Run 4."
- **2026-11-20 16:45 PST** (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`, F4 entry): "Yes, PWC
  follows range miner. Unless you have a other way to verify they both return the same results".
  The entry adds: "Whether either engine should honour the trial's sampled skip range is a separate
  RANGE-MINER question, out of scope here."
- **2026-11-20 21:32-21:35 PST** (same file): "Well range miner must be fixed none the less" and
  "Please do whatever method is the quickest and least recoding". It "supersedes the 16:45 entry
  above for S3-13 only … PWC keeps the trial's skip range".
- **2026-11-21 08:50 PST** (this request): quoted at the top.

**Beta**
- **X-VOLUME-LIMIT R1** (`docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md:79`):
  "**Status, already disclosed:** the builder emits no `skip_range`, and the worker reads its
  default `[0,16]` … This is faithful sampling of the current production behavior. It establishes
  no hit-rate effect; a selectable delivered-skip configuration compared with the baseline on
  unseen-draw pool hit rate would settle that separate question."
- **X-VOLUME-LIMIT R2** (`…_R2_BETA_REVIEW.md:80`): "**Recorded status:** delivered skip bounds
  remain absent from the builder, and the worker defaults to `[0,16]` … It samples current
  production behavior. This establishes no hit-rate consequence; a selectable delivered-skip
  configuration compared with the baseline on unseen-draw pool hit rate would settle that separate
  question."
- **X-PWC-STAGE3 R1** (`…/X-PWC-STAGE3_R1_BETA_REVIEW.md:115`): "The recorded constant-phase
  skip-range difference remains a confound for the owner-run engine comparison. Pin both arms to
  `[0, 16]` for that comparison or use the later miner fix. The hit-rate measurement is the pool's
  performance against `k/1000` on draws held out from the sieve."
- **Beta's callout on the hybrid phases (separate defect):** quoted in §2.3.

**Claude Code reports**
- **The survivor-volume report** (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:360-370`), F-1:
  "**phases 1–2 always scan skip 0–16, whatever was sampled.** … **What would settle the hit-rate
  question** (Beta Q1): a flag-selectable delivered-skip configuration, compared with the baseline
  on unseen-draw pool hit rate."

**NOT FOUND:**
- a TB_RULING deciding the constant-phase skip range;
- a `docs/LEADS.md` entry for F-1 (keyword search only).

## 6. What this record does not settle

- **Hit rate:** whether delivering the range changes pool hit rate against `k/1000` on unseen draws.
  This is the measurement in the summary. Nothing in the repository has run it.
- **Survivor counts:** phase 1–2 counts at a delivered 6–99 (§4.3).
- **Step 6:** whether a per-seed skip reaches prediction generation on the miner route (§4.4).
- **The bounds:** where the `skip_max` bound of 250 and the 10/10 floors come from (§2.4).
- **Whose fix this is:** this record recommends nothing and removes nothing. Per CLAUDE.md
  yardstick step 4, any alternative is proposed as a flag-selectable configuration with the current
  algorithm kept as the baseline. The options are in `SCOPE_MINER_SKIP_RANGE_FIX.md`.

## 7. Beta opinion X-SKIP-F1 R1

- **Where it came from:** relayed 2026-11-21 with the owner's approval ("please", 09:25 PST) for an
  **opinion, not a ruling**. One round. Verdict `REVIEW RECEIVED` (exit 0).
- **Reply:** `docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md`, sha256
  `a28b0bfef1bd0387d6ebc7acfb53e00414bc95ce4f525d325b55324ee465971f`. Read the file itself; the
  quotations below are excerpts.

**Beta's four answers, verbatim** (positions table `:25-30`):

| Q | Position | Beta, verbatim |
|---|---|---|
| Q1 design reading | NO OBJECTION | "I found no standing design decision to force `[0, 16]` for all constant trials. … The Phase-6 sampler harness explicitly inserts its own range, which accounts for the submission's claim." (`:36`) |
| Q2 direct fix or flag | SUGGESTION | "I favor a run-selectable delivered-range option with `[0, 16]` as the baseline through the comparison." (`:40`) |
| Q3 before or after Run 4 | SUGGESTION | "Keep Run 4 as the historical operational baseline and take up F-1 as a separate reviewed change after PWC Stage 4. Stage 4 can compare both engines with the trial pinned to `[0, 16]`." (`:46`) |
| Q4 operational effects | SUGGESTION | Five points (`:50-55`): compute and lease, "`[6, 99]` tries 94 hypotheses per seed versus 17, or 5.53× … 65,283 versus 1,972 source-level advances per seed (33.1×)"; stripe sizing and VRAM; the volume gate, "measure each intended geometry and interval before relying on the cap decision"; `best_skips`, "Validate `0 ≤ skip_min ≤ skip_max ≤ 255` at the common assignment boundary"; historical studies, "Start the delivered-range evaluation under a fresh study identity"; deployment, "the coordinator is also in Gate 12's rig parity set" |

**The two items this record had marked [UNVERIFIED], as Beta left them:**
- **An out-of-range warm-start pin (§4.2): resolved.** "In a local in-memory check, Optuna 4.4.0
  returned a fixed enqueued `skip_max=256` from `suggest_int(..., 10, 250)` with an out-of-range
  warning … A winning skip of 256 would serialize as 0 in the one-byte output. I did not execute
  the launcher or the kernel with that pin." (`:53`)
- **A per-seed skip at Step 6 (§4.4): resolved in code, not on a run.** Beta read the code path:
  "Step 6 looks for `skip`, `metadata.skip` or `features.skip_hypothesis`. The range-only fixture
  used the fallback as expected." On a real run, "The per-seed handoff on a real miner-to-pool run
  is **UNAVAILABLE**." (`:55`)

**Recommendations side by side — THE OWNER'S CHOICE:**

| | Alpha (original) | Beta (X-SKIP-F1 R1) |
|---|---|---|
| How | **(a)** a direct fix: RANGE-MINER delivers the trial's skip range to phases 1–2, as the design documents say | a **run-selectable** delivered-range option, with `[0, 16]` as the baseline through the comparison (`:40`) |
| When | **(b)** its own reviewed change, after PWC Stage 4 | the same: "a separate reviewed change after PWC Stage 4" (`:46`) |
| Judged by | **(c)** the same job at `[0, 16]` and at the trial's range; pool hit rate against `k/1000` on unseen draws; never true-seed recovery | the same, paired: "Report the paired hit-rate difference, each arm's hit rate against `k/1000`, and how often each arm produces a pool" (`:42`) |

**Alpha's revised recommendation.** Alpha changed (a) from a direct fix to a switch because of Beta's cost finding: 33.1×
the Java LCG source-level advances per seed at Run 4's geometry (X-SKIP-F1 R1 `:50`, Finding 2 `:60`). The revised
recommendation is to build the RANGE-MINER skip-range delivery as a switch with `[0, 16]` as the baseline, as its own
reviewed change after PWC Stage 4, settled by a paired hit-rate comparison on unseen draws.

**Owner's choice, 2026-11-21 09:54 PST, verbatim, in answer to that revised recommendation:** "sounds resonalble".
