# BACKLOG — tracked, not blocking Phase 7

**Purpose.** Everything below is known, deliberate to defer, and **not** a Phase-7 blocker. It is
written down so it is not rediscovered as a surprise finding in a later session, and so nobody
re-derives it from scratch at cost.

**Currency:** 2026-08-03, Phase 7 authorized. Every anchor in this file was re-read at source when
the file was written. An anchor is a claim with an expiry date — re-verify before acting.

**What is NOT here.** The hard Phase-7 prerequisites live in the skill's §8 approved sequence:
D6.2, D6.3, the `process_sharded` import gate, and 6-P2. This file is the *remainder*.

---

## 1. Unaudited chapters — the downstream half of the pipeline

**Chapters 3, 5, 6, 8 and 13 have had no scrutiny.** That is Step 3 scoring, Step 5 training,
Step 6 prediction, the PRNG registry, and the feedback loop — i.e. everything after the sieve.

Two facts set the prior for what an audit would find:

- **Chapter 1's audit found 9 of 41 claims accurate.** The base rate for an unaudited chapter in
  this project is not "mostly right with a few stale lines."
- **The one Step-3 finding we have surfaced sideways.** `forward_matches` / `reverse_matches` are
  absent from the Step-3 merge list — Beta called it *possibly the most consequential finding in
  the trace* — and it was found by an audit that was looking at something else. It was not found
  by anyone auditing Step 3, because nobody has.

`forward_matches` / `reverse_matches` are two of only four columns in the 22-array NPZ contract
that carry per-seed information (`seeds`, `forward_matches`, `reverse_matches`, `score`). They are
the **only independent per-seed sieve signal**. Their absence from the merge list needs a governed
schema decision; the miner keeps emitting both regardless, so nothing is being lost at the
producer — the loss is at the consumer.

**Implication worth stating plainly:** the audited half of the pipeline is the half that feeds the
unaudited half. Confidence in Step 2 does not transfer downstream.

---

## 2. Skip-output work — approved, sequenced after Phase 6

**The change.** Stop discarding `skip_sequences` at `window_optimizer_integration_final.py:147`,
and revive `skip_mean` / `skip_std` / `skip_entropy` — three of the five dead feature placeholders.
The producer already exists on the GPU. This requires **no kernel change**; only that the host stop
throwing the sequence away.

Verified at source (HEAD `09bbfbf`): `…final.py:140-151` builds each record as
`{'seed': seed, 'match_rate': rate}` and nothing else. The discard is not conditional and not
recent.

Lineage: the Oct-2025 output spec (`instructions.txt:1230-1245`) declares `skip_pattern` and
`pattern_stats: {mean_skip, variance, std_dev}` per survivor — the literal ancestor of the three
reviving features. This is a restoration, not an invention.

**The trap, and it is the whole difficulty.** Misses write a **fabricated `actual_skip`**.
Statistics computed naively over the returned sequence would therefore be statistics over
fabricated values. Every statistic must be taken **over valid hits only, behind an explicit
validity mask.**

**The open question that decides the shape of the work:** whether a hit mask reaches the host is
**[UNVERIFIED]**. If it does, this is a host-only change. If it does not, it becomes a kernel
change — a materially different piece of work with a different approval path. *Resolve this
before scoping, not during.*

**Naming, fixed:** `skip_search_*` for inputs, `observed_skip_*` for outputs. The whole
`skip_min`/`skip_max` confusion came from one pair of names doing two jobs at two pipeline stages.
Do not repeat it.

---

## 3. Sampler-comparison sequencing — a correction Beta issued against Alpha

The certifying four-phase TPE-vs-random comparison **cannot** be scheduled merely "after the
skip-output work." Alpha proposed that; Beta rejected it.

**Why.** The approved skip-output work (§2) retains observed sequences and restores three output
statistics. **It does not connect `skip_min` / `skip_max` to the hybrid kernels** — that is the
separate, unresolved **input-bound** interpretation, where the values die at `_hybrid_prefix`
(`range_miner_worker.py:177-193`). 22/22 constant kernels declare skip bounds; 0/22 hybrid do.

**Therefore the comparison waits for either:**
- hybrid search-input bounds with **defined effective semantics**; or
- an **explicitly phase-aware search space that does not pretend dead hybrid dimensions are
  active.**

Skip-output may proceed first. **Completing it alone does not remove the dead-dimension caveat.**
A sampler comparison run over a search space containing knobs connected to nothing measures the
sampler's behaviour in a void, and certifies nothing.

**Status quo:** TPE remains the production default *by status quo*. The five-seed run is a valid
constant-skip datapoint and useful directional evidence — **not** a certification of superiority,
and **not** authority for autonomous sampler selection.

---

## 4. Three `[WATCHER][RETRY]` log lines carrying the Chain C defect

`:1725`, `:1729`, `:1733`. The defect itself is fixed; these three call sites still emit the old
form. Cosmetic in effect, misleading in a soak log — which is exactly when someone will read them
and draw a wrong conclusion. Cheap to fix; not fixed yet.

---

## 5. Session-separated dataset authority

**Beta's ruling, and the part that is easy to misremember:** the combined publication was a
**provenance ruling**, not a finding that combined midday/evening records suit one PRNG model.

Midday and evening use **independently selected equipment** (draw procedures §II). There is
therefore **no evidentiary basis for advancing one PRNG state through interleaved records.**
Ordering is normative **within a session stream**; combined-container order carries **no
PRNG-advance meaning.**

Consequences already in force: the chronological-reorder migration was **cancelled**; combined-
session sequential sieve is **non-certifying and prohibited by default**; production
re-optimization is **per-session**.

What remains open is the *authority* question — which artifact is the session-scoped authority a
per-session run resolves against, given the pointer manifest currently addresses the combined
publication.

---

## 6. The non-terminating multi-stripe loopback — **[UNVERIFIED]**

A loopback run sized `total_seeds = 2 × miner_stripe_size` **does not terminate**; shards sit at
`staging_status='pending'`.

**What is established:** it reproduces with **no execution set frozen** and never enters the
admission-binding path, so it is independent of `eff6616`.

**What is NOT established:** whether this is a fixture limitation or a production defect. That
distinction is the whole question and it has not been answered.

**Why it is on this list rather than dismissed.** "Multi-stripe loopback does not terminate" sits
uncomfortably close to the §4.3 hang class — a trial that neither completes nor fails. That class
cost us an unbounded hang once. Better seen now than surfaced during a soak.

Beta has been asked (submission §6, ruling 3) whether this gets its own bounded investigation
before Phase 7. **Awaiting ruling.**

---

## 7. `dataset_provenance/*.json` never pruned

Same class as D6.3 — unbounded growth of run-scoped state with no retention policy. Newly found,
not yet scoped. Beta's D6.3 constraint applies by analogy and should be assumed to govern here
too: **never remove active, unresolved or audit-retained state merely for exceeding an age or
count threshold.**

---

## 8. Two read-only audits Beta wants before Phase 7

Both are **investigations, not repairs. No fix is authorised for either.**

### 8.1 `java_lcg_cpu` non-zero-skip mismatch

`prng_registry.py:170-183` applies skip **once before generating**; the kernel applies it
**between every draw** (`:987-989`). **They agree only at `skip=0`.**

Two call sites pass **non-zero** skip into that CPU path, both re-read at source:
- `survivor_scorer.py:124` — `self._cpu_func(seed=int(seed), n=n, skip=skip)`
- `full_scoring_worker.py:305` — `prng_func(seed, n_holdout, skip=offset)`, where `offset` is
  `train_history_len` and therefore routinely non-zero

**The audit question is reachability**: do these sites receive non-zero skip in practice, and if
so what does the resulting mismatch corrupt.

**Wall C caution, and the reason this matters beyond a local bug:** building the known-answer
reference on `java_lcg_cpu` would validate the **wrong semantics** — in the very deliverable whose
purpose is catching semantic error.

*Context: Michael reports all 44 PRNGs were validated through the sieves during pipeline
development — constant forward/reverse and hybrid variable-skip. An inventory is establishing what
exists before anything is scoped as new. §0.4's standing rule applies: absence of a working
implementation is not evidence of absent intent.*

### 8.2 Sampler provenance guard

`run_optimization()` trusts caller-supplied `sampler_class` / `sampler_module` / `optuna_version`
and **does not verify them against the actual sampler object.**

**Nothing already submitted is invalidated** — the existing TPE and Random wrappers are correctly
labelled. The exposure is forward-looking: **a fail-before-study guard is required before direct
use of the neutral core, or registration of any additional sampler.**

Note that `sampler` and `sampler_metadata` are already **required and keyword-only with no
default** — deliberately, so a caller cannot get TPE by omission and then report the run as
something else. **An unlabelled run is not a control.** The guard completes that intent; it does
not introduce it.

---

## 9. Small, verified, unfixed

- **`.gitignore:42` dead negation.** Line 41 is `*.json`; line 42 is
  `!config_*.json        # Keep config JSONs (safe & important)`. **`.gitignore` has no
  trailing-comment syntax** — a `#` is only a comment when it is the first character of the line.
  The pattern is therefore the entire line including spaces and comment text, and it negates
  nothing. Lines 43-44 (`!*_config.json`, `!schema_*.json`) have no trailing comment and do work.
  *This is why the `.json` extension is load-bearing for published dataset artifacts, and why they
  must not be named `*_config.json` or `schema_*.json`.*
- **The CA draw-procedures PDF is not in the repo.** It is the cited source for the physical model
  behind skip (§0.4 of the skill — pre-test sessions, per-session equipment selection, evening
  multi-game draws). Citation status is `UNAVAILABLE` until it lands. Everything derived from it is
  currently uncheckable by anyone reading only the repository.
- **Two doc-generator defects** — carried forward, not re-characterised here.
- **Chapter 1 §17.3 has 14 open items; Chapter 2 §14.3 has ten.** Enumerated in the chapters
  themselves; both chapters are closed as *verified and bounded*, which is not the same as
  *finished*.

---

## 10. Standing reminders that keep costing us

- **The repository is not the system (VIR-6).** systemd units, cron, host config and deployed
  uncommitted files are invisible to every repo-scoped gate. Alpha once reported "no scraper
  invoker exists" from a clone while an enabled boot-triggered unit sat on the host.
- **Gitignored files are invisible to every repo-scoped search.** `agent_manifests/trse.json` —
  the file *causing* TRSE F1 — had no git history at all until `93918f5`.
- **A keyword hit is not a finding until the surrounding text is read.** Four absence claims were
  falsified in one session; the last was made after a grep that **reached the exact line and did
  not read it.**
- **Cited is not read.** F6's specification sat in `TRSE_INTEGRATION_PLAN_S121.md`, tracked, cited
  repeatedly, unopened.

---

## 11. `_RusageChildrenSampler` measures the wrong thing — G-RSS passes by luck

`tests/test_s172_phase5_d5_process_sharded.py:2107-2119` reads
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, whose docstring calls it *"the maximum of any
SINGLE reaped child."* It is a **process-lifetime high-water mark over every child the process has
ever reaped — not scoped to its `with` block.**

Measured on VM101: `0 MiB` → trivial child `10 MiB` → **one torch-importing child `378 MiB`** →
another trivial child **still `378 MiB`.** The mark persists.

So `G-RSS` silently depends on **no earlier child in the whole D5 run exceeding its own ~339 MiB
tree-sum.** When the import gate first sat beside `G-NO-GPU`, G-RSS red and **mutant M8 survived** —
deterministically, not flakily. Contained by moving the gate after `G-MUTANTS`.

**Any future D5 arm that reaps a large child reds it the same way.** Scope-correct fix: record
`ru_maxrss` on `__enter__` and compare the delta. **Flagged, not actioned** — it edits an existing
D4/D5 arm.

---

## 12. D3.0-B — OPEN and requiring completion; it NARROWS what Phase 6 certified

A stated Phase-6 certification prerequisite that was never completed; Phase 6 certified anyway.
The defect is still live, re-read at source — `convert_survivors_to_binary.py:184`:

```python
encode_prng_type(s.get('prng_type', s.get('prng_base', 'java_lcg')))
```

A record carrying **neither** `prng_type` **nor** `prng_base` is **fabricated as `'java_lcg'`**
instead of failing closed — while the canonical resolver already provides the fail-closed
behaviour.

**Beta ruled 2026-08-02: OPEN and REQUIRES COMPLETION.** *Waived* and *superseded* were both
**rejected** — REV3 made it mandatory, the defect remains executable, divergent encoding tables
persist in dormant-but-executable writers **and patch scripts**, and no ruling ever removed the
prerequisite. **Beta recorded its own Phase-6 certification as a governance error** for omitting
it, and disclosed that unprompted.

**The certification scope is narrower than "Phase 6 is certified."** Phase 6 is certified for the
demonstrated **miner/finalizer path** — Wall A used the miner coordinator, Phase-5 assembly, the
D3.5 finalizer, direct 22-array validation and Step-2/Step-3 consumption, and **never invoked
`convert_survivors_to_binary.py`.** **Legacy conversion and dormant legacy-writer surfaces are
UNCERTIFIED.** No Wall A/B rerun is required.

**Do not invoke the legacy converter until D3.0-B closes.**

**Bounded scope when it is done:** canonical fail-closed resolver replacing missing-identity
defaults · preserve valid `prng_type` precedence and valid `prng_base` fallback · reject records
carrying neither · **remove or hard-retire divergent executable encoding tables, including
rerunnable patch scripts that could reinstall them** · behavioural gates and mutants for missing
identity, unknown identity, and reintroduced `java_lcg` defaulting.

**Does NOT block the miner-backed Phase-7 soak** — the soak does not invoke the legacy writer —
and 6-P2 remains independent. **Not scoped for implementation.**

Alpha's original position stands and the ruling vindicated it: **Alpha did not propose a fix**,
because it touches the legacy writer and, if it genuinely should have blocked certification,
closing it quietly is the wrong move. That was the right call — Beta's answer was neither of the
two exits Alpha's notice offered.

Sources, all re-read: `docs/TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md` (the ruling request),
`docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1 (Beta's disposition, accepted in full), skill §2.18.

---

## 13. NP2 checkpoint transaction design — NEW, separate work

D6.2 is certified for **`n_parallel == 1` only.** Concurrent partition writers cannot safely share
the present two-member checkpoint pair. **`resume_checkpoint` + `n_parallel > 1` is refused as the
first executable statement of `optimize_window`**, and **Phase 7 pins `n_parallel=1`** until this
lands. Not scoped.

---

## 14. `2019-01-25` is evening-only

A single-session date in the modern era. Of 1,040 single-session dates in the dataset, **1,038 are
2000-2002** — the era before CA Daily 3 had a midday draw — leaving 2019 with one and 2026 with one.
The 2026 case is the scrape's truncation point. **`2019-01-25` is a genuine anomaly**: a real
cancellation, a source gap, or a scrape defect. One record in 18,068. **Not 6-P2's work.**

---

## 15. Step 3's output validation floor is three contracts stale

`run_step3_full_scoring.sh:475-478`, the "Phase 6: Validate Output" block — re-read at source:

```
    # Check feature count (should be 50)
    feature_count = len(sample.get('features', {}))
    if feature_count < 46:
        print(f"❌ VALIDATION FAILED: Only {feature_count} features found (expected 46+)")
```

**The live contract is 91 extracted / 89 trained.** A run emitting **46** features passes this
wall and prints `✅ VALIDATION PASSED`. The comment guarding the test names a *third* figure —
**50** — that matches neither the test below it nor the contract. **The wall is set three
contracts behind the code it guards**, and a silent 45-feature collapse to 46 is the only failure
it can still catch.

The information needed to set the floor correctly is already in hand one phase earlier in the same
script: `:426-431` splits `global_*` from per-seed features and prints per-seed, global and total
counts. The aggregation phase knows the shape; the validation phase does not use it.

Not blocking; not fixed. Source: the Step-3 script read, 2026-08-03; line numbers re-verified at
HEAD `d99923b`.

---

## 16. `full_scoring.json` declares 26 GPUs; the frozen Phase-7 set is 25

`agent_manifests/full_scoring.json:102-110` declares `parallel_workers` with `"max": 26`,
`"default": 26`, and the note *"Use full cluster (26 GPUs) for maximum throughput."*

The frozen Phase-7 execution set is **25** — `set_id bea580e764905a0d9485d2688be5841cc95f16e161…`
(`bea580e76490`), 25 identities requested and 25 admitted, `clamped = False`, **25 by construction
and not by clamp**. Owner-ruled, Beta-ratified.

**No execution consequence, and the wiring evidence is stronger than the manifest's own note.**
`parallel_workers` lives in `parameter_bounds`, **not** in `default_params` — so WATCHER's
step-scoped filter (`agents/watcher_agent.py:1290-1314`, `if key in declared`) could not pass it
even if something wanted it. No `args_map` in any of the manifest's three actions references it,
and `run_step3_full_scoring.sh` never mentions the string at all (grep: no hits). **This is
documentation drift, not a dead wiring path** — there is no path.

*(Note for whoever fixes it: the manifest's `_note_default_params` at `:141` lists `prng_type`,
`mod`, `batch_size`, `jobs_file`, `output_file` as documentation-only. It does **not** cover
`parallel_workers`, which is a different field in a different block. Do not cite it as the
authority here.)*

**Fourth place carrying a stale 25-vs-26 figure**, and the enumeration matters because the fourth
looked like the last two: `PROJECT_FILE_CATALOG.md` and `PIPELINE_BEHAVIOUR_MODEL.md` were both
corrected (`f8cb1c5`, `c4917a8`); `distributed_config.json` now totals **25** (`localhost.gpu_count`
2 → 1, `f255912`); and **`ml_coordinator_config.json` totals a live 26 and correctly stays** — it
declares hardware, not the execution set. The divergence between those two config files is recorded
as **D17** in `PIPELINE_BEHAVIOUR_MODEL.md:1193`. Not blocking; not fixed.

---

## 17. A skill revision lives in three places, and committing updates one

| copy | who loads it | how it updates |
|---|---|---|
| `docs/TFM_PROJECT_FACTS_SKILL.md` | nobody at runtime — the tracked source | commit + dual-push |
| `~/.claude/skills/tfm-project-facts/SKILL.md` | **Claude Code**, on invocation | **manual `cp`** |
| the Settings upload | **new chat sessions**, at session start | **manual re-upload** |

**Nothing warns you when they diverge, and they diverge silently.**

On **2026-08-03** the tracked copy reached **v13** while the installed copy still held **v6** — last
touched 00:22 that day, **before the entire day's work** — and Settings held **v11**. **Thirteen
revisions, and not one had reached a runtime copy.** Every correction made that day protected
nothing until the copies were fixed by hand.

**A running chat session cannot be updated at all** — its copy is fixed at session start. After any
revision that matters, start a fresh session.

The four-step completion rule (commit + dual-push · back up then `cp` the installed copy · re-upload
to Settings · verify in a fresh session by printing the Currency line and the §0.6 heading verbatim)
lives in the skill's §7 working agreements. **The currency line exists to make this drift visible;
it is the only signal there is.**

**This entry exists so the failure is findable from the backlog too** — a reader who has the stale
skill loaded is exactly the reader who cannot find the warning inside it.

---

## 18. `chapter_13_triggers.py` reaches Step 3 outside `--end-step`

The soak's `--end-step 1` bounds WATCHER's pipeline. It does **not** bound Chapter 13, which has two
routes to Step 3 — **both requiring a human action**, so neither is a live risk:

- **Standalone.** `execute_standalone` (`:630`) carries **its own** `STEP_SCRIPTS` dict (`:644`),
  where `3: "run_step3_full_scoring.sh"` (`:647`), invoked via `subprocess.run` (`:672`). Sole
  caller is the CLI `--execute` flag at `:932` (`--steps` defaults to `[3, 5, 6]`, `:885`).
- **Learning loop.** `execute_learning_loop` (`:579`) defaults to `steps = [3, 5, 6]` (`:597`) and
  calls `self.watcher_agent.run_pipeline(start_step, end_step, params)` at `:616` with
  `start=min(steps)`, `end=max(steps)` — **a fresh `run_pipeline` with its own bounds. A soak's
  `--end-step 1` does not constrain it.**

Note the standalone dict's step 2 is `run_scorer_meta_optimizer.sh` — the script that invokes the
TB-prohibited converter and `mv`s a regular file onto the D3.5 finalizer-owned symlink. A
`--steps 2 …` standalone invocation reaches it directly, with no `--end-step` anywhere in the path.

**Standing operational rule: no Chapter-13 retrain approval and no learning-loop invocation while a
soak is running.** Not a code defect and nothing to fix — an operational constraint that has to be
written down somewhere a soak operator will look. Line numbers re-verified at HEAD `d99923b`.

---

## 19. `.s172_accumulator/generations/` is durable data plane with no backup policy

*Added 2026-08-17 from the Gate-12 Attempt-9 acceptance ruling
(`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`). Ruled real, ruled non-blocking.*

Since the publication-path change, `bidirectional_survivors_binary.npz` and its siblings are
**symlinks into a per-generation accumulator**, not tracked regular files. Beta's ruling makes
the consequence explicit:

> `.s172_accumulator/generations/` is now part of the durable data plane. It needs
> backup/recovery policy appropriate to an authenticated generation store. **Git is no longer
> that backup.**

Two dispositions, both Beta-ruled:

- **Do NOT put the NPZ payloads back into git** merely to satisfy the historical rule that
  applied when `binary.npz` was a tracked regular file. The clean-tree gates passed; the
  post-run untracked paths are the run's own outputs, written after publication.
- The **stale `.gitignore` negation** left over from the tracked-file era can be removed as
  hygiene when convenient. Not a Gate-12 issue, not urgent.

What needs designing (unowned, unscheduled): retention count per generation, off-VM101 copy
target (the ser8 `~/Downloads/` forensic-archive convention is the existing precedent — the
Attempt-9 bundle `gate12_attempt9_forensic_bundle_20260817.tar.gz`,
sha256 `583fbab3f4f7772f5405f302dbea596e8303a71420a0b2445149025470743fa2`, lives there),
integrity verification on restore (the accumulator is an *authenticated* store — a backup
that can't prove generation identity on restore is not a backup of it), and what "recovery"
means for a partially-restored generation directory. Until that exists, the only copies of
certified survivor generations are the live VM101 filesystem.

---

## DEP-ABI-V2-NPZ-SEMANTICS — audit dependency, recorded by TB ruling (Brief-I production-shape failure, 2026-08-22)

**Status: RECORDED AS A DEPENDENCY. No 22-array amendment is pre-authorized.**

**The settled v1 semantics.** Canonical array 4 `offset` is a **legacy wire name with exactly
one post-F-4 meaning: it IS the window anchor.** It is **never** the generator phase, at any
phase value. Generator phase is independently represented in versioned generation metadata and
never enters array 4. The name is frozen by the 22-array contract and does not change; only the
source of its value was corrected (`utils/canonical_records.py:217`).

**The dependency.** `DEP-ABI-V2` (§2.53) is the separate kernel/parity certification cycle that
would make a nonzero `generator_phase` production-permissible on the four no-phase forward
hybrids.

> **Before nonzero phase becomes production-permissible, re-audit EVERY consumer of canonical
> `offset` and prove each reads that field EXCLUSIVELY as the window anchor, while
> `generator_phase` remains independently available from versioned generation metadata.**

**Why this is not optional — and note the correction.** Alpha originally argued that array 4 is
coherent *only* because `generator_phase == 0`. **Beta overruled that rationale.** Array 4 has a
deliberately legacy **wire name** but a single post-F-4 meaning at every phase value; the
misleading token does not fuse the two quantities **so long as no consumer treats it as both
concepts**. Phase is independently and durably available from versioned generation metadata.

**Therefore ABI-v2 does not automatically resurrect F-4.** It would resurrect F-4 only if some
consumer again interprets array 4 as *both* anchor and phase — which is exactly what this audit
exists to rule out. If every downstream consumer can correctly obtain phase from metadata, the
existing 22 arrays may remain unchanged even under nonzero phase. Only a governed downstream
contract that genuinely requires phase *inside* the canonical array body would trigger a separate
array-contract amendment, and **no such amendment is pre-authorized.**

**Known consumers to start from (not a complete census — the audit must derive its own):**
`utils/canonical_records.py` · `utils/canonical_arrays.py` (deliberate duplicate field list) ·
`utils/run_finalizer.py` `CANONICAL_ARRAY_CONTRACT` · `utils/checkpoint_d6_2.py` ·
`survivor_scorer.py` merge list · Step-3 / Step-6 readers.

**Precedent for how it is found:** the Brief-I defect was a single unmigrated consumer of the
retired context key, on the production path, in a file the brief never opened, behind 25 green
gates. **A gate proving one consumer migrated is not evidence that every consumer did.**

---

## 20. Coordinator ingress is bounded by COUNT and unbounded in BYTES — allocation-limited

**Recorded 2026-08-27. Design observation, flagged for Beta. No fix proposed, none authorized.**
Full arithmetic: `S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §4.4 · register entry
`LEADS.md` **L-2**.

`miner/range_miner_coordinator.py:9602` — `inbound: _queue.Queue = _queue.Queue(maxsize=1024)`.
The connection reader decodes each frame and puts the **decoded message object** on this queue
(`:11284`). The bound is 1,024 *entries*; the byte weight of an entry is linear in the survivor
rate and bounded nowhere. **Every volume-aware bound in the system — `staging_high_water_files`,
`staging_high_water_bytes`, the derived deferred bound — sits downstream of the one place that
has none.**

**How it surfaced.** By elimination, characterizing what limits survivor volume after run 3
(13,146,485 phase-3 survivors, 1,597× attempt 9). The two candidate retention ceilings were both
withdrawn on measurement: the **file-count bound is survivor-independent** (attempt 3 with 774
survivors reached 91.9% of it; run 3 with 13.1M reached 81.9%), and the **16 GiB byte bound peaks
at ~61% utilisation** from inside the Optuna-reachable space. Ingress is what was left.

**⚠ IT IS AN ALLOCATION LIMIT, NOT AN ARCHITECTURAL ONE.** VM101 is a Proxmox VM; the Zeus host
has 64 GB. At the current allocation (`MemTotal 15,924.8 MiB`, `SwapTotal 0`, live-read) the
computed exhaustion point `r_crit = 0.0602–0.0688` **straddles the search space's provable
reachable maximum `r_max = 0.0642`**. Roughly **17 GiB** clears a full queue at `r_max`; ~32 GiB
leaves 2× headroom. **Raising the allocation moves the ceiling above the reachable maximum.**

**Two operational constraints on doing that:**
- **Not before run 4.** Run 4 is the Brief I acceptance run; changing VM101's memory first puts a
  second variable into it. **Wait until Brief I closes.**
- **Swap is a separate lever.** `SwapTotal = 0` is what makes the failure mode a hard OOM rather
  than degradation telemetry could catch. Adding swap changes the failure *mode*, not the ceiling.

**Why it is still worth Beta's attention with a bigger box available:** ingress admission control
**cannot see volume at all**, so the safe operating region is a function of an ungoverned resource
(host RAM allocation) rather than of any governed bound. Raising the allocation moves the ceiling;
it does not give the system a way to know where the ceiling is.

**Status: OPEN.** Computed, never observed — the four assumptions are enumerated in §4.4 and in
L-2. The cheapest thing that would turn it from computed into observed is a **coordinator RSS
sampler on the next run**; `[S172-BP]` records queue *occupancy* (550/1024 in run 3) but no byte
or RSS series exists. That needs no fleet of its own — it rides whatever runs next.

---

## 21. float32/float64 survivor-filter seam — effective M is set by the HOST

**Recorded 2026-08-27. Defect. Repair NOT authorized and NOT proposed** — either side could be
made to match the other and the two choices yield different survivor populations, which makes it
a semantics decision. Register entry `LEADS.md` **L-1**; derivation
`S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` §2d.

The kernel filters in float32 (`if (best_match_rate >= threshold)`); the host re-filters the same
value in float64 (`miner/range_miner_worker.py:1281`). When `k·τ` is an exact integer `m` and
`m/k` is not exactly representable in binary32, the widened float32 rate lands just below the
float64 threshold and **the host rejects seeds the kernel admitted**, silently raising the
effective match requirement to `m+1`.

**Measured** (production kernel, 2²⁴ seeds, k=20, τ=0.35 → `7/20 = 0.34999999403953552`):
**9,575 kernel survivors → 1,513 host survivors, a 6.3× population change.** Three of 64 grid
cells affected (`k=20/τ=0.35`, `k=20/τ=0.45`, `k=10/τ=0.70`); every other exact-`kτ` cell checked
and consistent, so the residue is fully explained.

**What it means for correctness:** the survivor population reaching Step 2 and the 22-array NPZ
depends on a floating-point representability accident, which makes populations at those boundaries
**non-reproducible across any change to that comparison** — a dtype change, a threshold
quantization, a refactor comparing float32 on both sides, or a move of the post-filter. The
affected thresholds are round decimals (0.35, 0.40, 0.45, 0.50, 0.70, 0.75), i.e. exactly what
Optuna's `round(…, 2)` canonicalization produces. **Status: OPEN**, with three follow-ups
registered in L-1, including whether it wants one ruling together with the open §2.36 Optuna
raw→canonical quantization item.

---

## 22. Privileged internal seams `_operator_pin_params` / `_pin_bundle` — governance-gated

**Recorded 2026-08-28. Standing constraint adopted by Team Beta in
`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` §1. Binding. No code required now.**

> `_operator_pin_params` and the internal `_pin_bundle` are **privileged internal seams. No new
> production caller may populate either without an explicit governance review.**

**Why it is load-bearing.** These two names carry *operator authority*, not data. The whole R1
correction was that authority must be provable by ORIGIN rather than inferred from the presence of
a complete bundle in ordinary `params`. Today exactly one production site populates each:

| seam | sole production populator |
|---|---|
| `_operator_pin_params` | the `--run-pipeline` CLI branch of `agents/watcher_agent.py`, via `split_operator_pin_params()` |
| `_pin_bundle` | `run_pipeline()`, from the authority channel, threaded to that invocation's Step-1 `run_step()` |

A second populator added later — however reasonable locally — silently recreates the R1 defect: a
programmatic caller acquiring explicit-operator provenance. **`G-ORIGIN` and mutant `M4` in
`tests/test_s172_run4_routing_patch.py` are the standing detectors**; M4 is precisely "capture
reverted to ordinary `params`". A new populator that routes correctly would still pass G-CHAIN and
G-UNPINNED-IDENTICAL, so the review is the control, not the suite.

**Applies to** any new caller in `agents/`, `chapter_13_triggers.py`, the selfplay/dispatch paths,
`step_runner`, or a future scheduler. `step_runner`'s separate dispatch surface is **not** part of
the certified Run-4 route (ruling §8) and must not be wired to these seams to make it one.

---

## 23. Stale root-level `watcher_agent.py` — wrong-file-edit hazard

**Recorded 2026-08-28. Carried unchanged from the R2 approval's recorded-but-unrepaired list
(ruling §8). Does not block the R2 commit. No fix proposed, none authorized.**

Two files share the name. The root copy is dead and badly stale:

| path | size | mtime | state |
|---|---|---|---|
| `agents/watcher_agent.py` | 171,520 B | 2026-08-28 | **LIVE.** The certified R2 target, sha256 `d14bcb3b…` |
| `watcher_agent.py` (repo root) | 72,042 B | 2026-04-24 | **STALE.** Four months behind |

**Measured drift, not assumed:** the root copy contains **zero** occurrences of
`_operator_pin_params` or `STEP1_EXPLICIT_PIN_KEYS`, and its pipeline loop still reads
`results = self.run_step(step, params)` at `watcher_agent.py:1547` — the pre-patch call with no
`_pin_bundle` argument.

**The hazard is editing or importing the wrong one.** A patch applied to the root copy would land
in a file nothing executes, pass a naive `grep` of "watcher_agent.py", and produce a green-looking
session that changed nothing. The reverse — a tool resolving `import watcher_agent` to the root
copy — yields a WATCHER with no Run-4 routing at all and no error.

**Mitigation in force today is convention only:** every governance artifact in this workstream
names the path as `agents/watcher_agent.py`, never the bare filename. **Deletion is NOT proposed
here** — this repo's standing ruling on stale duplicates (2026-07-31) is to leave them alone
absent an explicit ruling, and that ruling is Beta's to make, not Alpha's.

**SKILL UPDATE NOTE (for whoever performs the next skill bump).** Both items above belong in the
skill's standing-rules section. Per §17 of this backlog a skill revision lives in **three places**
and committing updates only one, so this text is staged here rather than applied piecemeal:

- **SR-3 (proposed):** `_operator_pin_params` / `_pin_bundle` are privileged internal seams; a new
  production populator requires governance review. Detectors: `G-ORIGIN`, `M4`.
- **Verified-as-built fact:** `agents/watcher_agent.py` is the live WATCHER. Root-level
  `watcher_agent.py` is stale as of 2026-04-24 and must never be edited, imported, or cited.
