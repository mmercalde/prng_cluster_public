# CHAPTER_3_ALIGNMENT_AUDIT.md — REV1

**Read-only audit. NO FIX WAS MADE. Nothing was repaired, rewired or refactored.**

**Base:** `575378e` (`git pull` → *Already up to date*). Host: **VM101** as `michael`,
venv `~/venvs/torch` active for every command. Target document:
`docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` (958 lines, v4.2, last touched `05b0e6b`).

**Completion sentinel: `PASS`** — Q1–Q5 all answered from live source. Two sub-questions are
marked `UNVERIFIABLE` inside Q5 and are named as such; no question is `INCOMPLETE`.

---

## 0. Verdict in one paragraph

Chapter 3 describes a stage that **still executes** — `run_scorer_meta_optimizer.sh` is
`STEP_SCRIPTS[2]` and `generate_scorer_jobs.py` / `scorer_trial_worker.py` are both live in
its path — but it describes the **wrong version of it**. The worker is **v4.3**, not v4.2;
its objective function was replaced wholesale in a commit whose message is about moving docs.
Of the chapter's substantive technical content, the entire GPU-scoring and memory-batching
half (§8, §9, §14.2) describes code that was **deleted at v4.0** and is not in the file at
all. Two findings outrank the documentation question entirely: **the stage invokes the
TB-prohibited legacy converter**, and **it writes a regular file at a D3.5 finalizer-owned
symlink path**, which fails the next `finalize_run` closed. Both are reachable from WATCHER
step 2 and are live today because the accumulator is absent on this box.

---

## 1. FINDINGS — ordered by consequence

Scope key: **[repo]** verified from committed source only · **[live]** verified against live
host state (filesystem, execution).

---

### F1 — Step 2 invokes `convert_survivors_to_binary.py`, which TB prohibits **[repo+live]**

**Claim.** The Phase-7 soak's step 2 will run the legacy converter that D3.0-B leaves open.

**Anchors.**
- `agents/watcher_agent.py:390` — `STEP_SCRIPTS[2] = "run_scorer_meta_optimizer.sh"`.
- `run_scorer_meta_optimizer.sh:87` — `python3 convert_survivors_to_binary.py "$JSON_SOURCE" --output "$TMP_NPZ"`.
- Trigger conditions `:140` (`[ ! -f "$SURVIVORS" ]`) and `:144` (`[ "$JSON_SOURCE" -nt "$SURVIVORS" ]`).

**Live state on VM101 makes the conversion branch the one that runs.** There is no
`bidirectional_survivors_binary.npz` at repo root and **no `.s172_accumulator/` directory at
all** (`ls` → *No such file or directory*), so the D3.5 compatibility symlink does not exist,
`-f` is false, and `convert_to_npz` is entered unconditionally. Its input
`bidirectional_survivors.json` is present and is **2 bytes — literally `[]`**.

**What depends on it.** This is not a latent path. Any WATCHER run that reaches step 2 from
the current box state executes the prohibited converter as its first act. **No fix is
authorized here and none was made** — the disposition is Beta's.

---

### F2 — Step 2 writes a regular file at the D3.5 finalizer-owned symlink path **[repo]**

**Claim.** The same auto-conversion block places a regular file at
`bidirectional_survivors_binary.npz`, which the finalizer owns as a compatibility symlink and
which fails closed when a regular file appears there.

**Anchors.**
- `run_scorer_meta_optimizer.sh:97` — `mv "$TMP_NPZ" "$SURVIVORS"`, where `$SURVIVORS` is
  `bidirectional_survivors_binary.npz` at repo root (`:31`).
- `utils/run_finalizer.py:130` — `BINARY_NPZ_NAME = "bidirectional_survivors_binary.npz"`.
- `utils/run_finalizer.py:1400-1412` — for `ALL_NPZ_NAME`/`BINARY_NPZ_NAME`, if the alias
  exists and `not os.path.islink(alias)` → `PublicationError(... "something wrote outside the
  finalizer — failing closed rather than replacing it.")`.

`mv` uses `rename(2)`, which **replaces the symlink itself**, so this destroys the alias
rather than writing through it. Two consequences: the next `finalize_run` raises
`PublicationError` at publication, and the artifact whose SHA-256 is `artifact_sha256`
(`run_finalizer.py:136`, `CANONICAL_NPZ_NAME = BINARY_NPZ_NAME`) is replaced by a
`np.savez_compressed` product of a different producer (`convert_survivors_to_binary.py:201`).

**Aggravating.** `run_scorer_meta_optimizer.sh:120` then `scp`s that file to every node in
`distributed_config.json`, so the substitution propagates off-box.

**What depends on it.** Item 5's clean-tree/publication preflight is not the only wall this
trips; the finalizer's own fail-closed check is. In a soak, this surfaces **hours in, at
publication** — the exact failure mode §7 of the brief warns about, arriving from a different
direction than an untracked report file.

---

### F3 — The worker is **v4.3**; the chapter documents v4.2, and so does the file's own docstring **[repo]**

**Claim.** Chapter 3 §7.2's objective function is not the objective that runs.

**Live objective** — `scorer_trial_worker.py:439-443`:

```
objective = 0.70 * tanh(enrich) + 0.20 * coverage - 0.10 * size_penalty
```

where `enrich = log(p_sub / p_global)` over `skip_mode == 1` membership
(`scorer_trial_worker.py:421-426`).

**Chapter §7.2** documents `percentile(bidirectional_count) + 0.10 * median(intersection_ratio)`.
**Neither `bidirectional_count` nor `intersection_ratio` appears anywhere in the live
objective.** The live code says so explicitly at `:414-419`: *"bc_score (median
percentile-rank) is structurally dead: 79.2% of pool at bc>=11300 => any large subset has
constant median."*

**The file misdescribes itself too.** Its module docstring (`:3`), the `run_trial` docstring
(`:298`) and the "TB FORMULA (final v4.2)" block (`:311-322`) all still advertise the v4.2
formula that `:439` no longer computes. So a reader who checks the source docstring instead of
the source gets the same wrong answer the chapter gives.

**Verdict on the chapter here:** `FALSE`, not `STALE` — §7.2 presents executable code that
does not exist.

---

### F4 — The objective rewrite landed inside a commit about moving documentation **[repo]**

**Anchors.**
- `git log --oneline -S "enrichment objective" -- scorer_trial_worker.py` → **exactly one
  commit: `ca975f8`.**
- `ca975f8` subject: **`chore(S109): move 58 stray docs from root to docs/`**; body: *"Session
  changelogs, proposals, addenda, specs, and guides were scattered in the project root.
  Consolidated into docs/ directory."*
- `git show ca975f8 --numstat -- scorer_trial_worker.py` → **46 insertions / 45 deletions**.

The chapter was updated to v4.2 at `05b0e6b` and was **correct when written**. The objective
changed one commit later under a message that mentions only doc consolidation. This is the
same failure class as the threshold regression catalogued as §2.7 instance 2 — *a behavioural
change carried by a commit whose message never names it*. It is the mechanism by which this
chapter went stale, and it is worth recording separately from the staleness itself.

---

### F5 — `UnboundLocalError` on every trial when the survivor pool is at or below `sample_size` **[live, execution-proven]**

**Claim.** `run_trial` assigns `sm_arr` only in the sampling branch; the no-sampling branch
falls through to a reference of it.

**Anchors.** `scorer_trial_worker.py:356-366` (if-branch assigns `sm_arr`, and does so
**twice** — `:365` and `:366` are duplicate lines) vs `:369-375` (else-branch assigns
`bc_arr`/`ir_arr`/`fwd_arr`/`rev_arr`/`tn_arr` and **not** `sm_arr`), consumed at `:421`.

**Execution proof.** Live `run_trial` called with synthetic in-memory arrays (no files, no
GPU, no NPZ, no converter, no pipeline):

```
--- CASE A: n_seeds(2000) <= sample_size(50000)  [else-branch] ---
RESULT: UnboundLocalError: local variable 'sm_arr' referenced before assignment
--- CASE B: n_seeds(2000) > sample_size(500)     [if-branch] ---
RESULT: returned objective=0.06947668617266764
```

**Reachability.** `sample_size` reaches the worker from the job JSON
(`generate_scorer_jobs.py:69`), whose value is `--sample-size`: default **25000** in the
generator (`:85`), **5000** from the shell (`run_scorer_meta_optimizer.sh:171`), **450** in the
manifest. Production pools of ~742K seeds take the if-branch, so this does not fire today. Any
run against a pool at or below the effective sample size takes the else-branch, and then
**every trial** fails to `status: "error"`, `accuracy: -inf` (`:625-628`) and the study still
produces an `optimal_scorer_config.json` from an arbitrary trial.

---

### F6 — The objective is structurally blind to 7 of the 11 sampled dimensions **[repo]**

Only `residue_mod_1/2/3` and `max_offset` reach the objective, via the k-of-3 residue mask
(`scorer_trial_worker.py:380-391`).

| sampled parameter | reaches objective? | anchor |
|---|---|---|
| `residue_mod_1/2/3`, `max_offset` | **yes** | `:380-391` |
| `temporal_window_size` | **no** — read into `tw_size` at `:347`, never referenced again | `:347` |
| `temporal_num_windows` | **no** — never read by the worker | — |
| `min_confidence_threshold` | **no** — never read by the worker | — |
| `hidden_layers`, `dropout`, `learning_rate`, `batch_size` | **no** — never read by the worker | — |

Also dead in the same function: `bc_arr`, `ir_arr`, `fwd_arr`, `rev_arr` are all computed at
`:360-374` and never used; `LAMBDA_SIZE` (`:339`) and `IR_WEIGHT` (`:341`) are defined and never
used. These are v4.2 remnants the v4.3 patch left behind.

**Consequence.** `optimal_scorer_config.json` is `study.best_params`
(`run_scorer_meta_optimizer.sh:306-307`). The winning trial is selected on a score that cannot
see 7 of its 11 values, so those seven are **whatever the TPE sampler happened to draw for the
trial that won on the other four.** The live file confirms this shape today
(`optimal_scorer_config.json`, 11 keys, mtime 2026-03-11).

This is the §0.5 dead-dimension pattern — *"the sampler steers a knob connected to nothing"* —
at the search-space level rather than the kernel level. **Reported, not fixed.**

---

### F7 — Chapter §6.1's "Nothing to remove from the search space" is FALSE for four parameters **[repo]**

Chapter §6.1's S108 note claims `hidden_layers` + `dropout` + `learning_rate` + `batch_size`
"feed Step 5 anti_overfit_trial_worker."

**They do not.** Step 5 samples its own from an independent Optuna study and never opens
`optimal_scorer_config.json`:

- `/bin/grep -c optimal_scorer_config` → **0** in each of `generate_anti_overfit_jobs.py`,
  `anti_overfit_trial_worker.py`, `meta_prediction_optimizer_anti_overfit.py`,
  `run_anti_overfit_optimizer.sh`.
- `generate_anti_overfit_jobs.py:29-43` samples its own `hidden_layers`, `dropout`,
  `learning_rate`, `batch_size`.

The value sets are not even compatible: Step 2 samples `hidden_layers` as `'128_64'`
(`generate_scorer_jobs.py:63`); Step 5 expects a JSON list-string `'[128, 64]'`
(`generate_anti_overfit_jobs.py:30-36`, parsed at `anti_overfit_trial_worker.py:145-147`).
Ranges differ too (`learning_rate` 1e-4–1e-2 vs 1e-5–1e-2; `batch_size` [32,64,128] vs
[16,32,64,128]).

**The other half of §6.1's note is ACCURATE.** `residue_mod_1/2/3`, `temporal_window_size`,
`temporal_num_windows` and `min_confidence_threshold` **are** consumed by Step 3's
`SurvivorScorer` — `survivor_scorer.py:97-111`, `:396-397`, `:670-672`, `:704`. Those seven
have a real consumer; they simply have no objective signal (F6).

---

### F8 — The v4.3 objective optimizes toward the hybrid/variable-skip population **[repo]** — contact point only

`skip_mode == 1` decodes to **`'variable'`** — `utils/prng_encoding.py:37`,
`SKIP_MODE_ENCODING = {'constant': 0, 'variable': 1}`. The enrichment term
(`scorer_trial_worker.py:421-426`) therefore rewards residue masks that **preferentially select
hybrid (variable-skip) survivors**, described in the code comment as *"the skip_mode==1
minority island (8.1% of pool, structurally distinct)"* (`:417-419`).

That is the population produced by the path where sampled `skip_min`/`skip_max` die at
`_hybrid_prefix` and `expected_skip` is hardcoded (§2.7 instance 4, OPEN). The Step-2 objective
is now the only place in the pipeline that actively steers *toward* it.

**Noted as a contact point and left alone**, exactly as the brief directs for the
`java_lcg_cpu` case. It needs a governed decision, not a patch.

---

### F9 — Step 2 introduces three fleet surfaces, none of them among Beta's six, none a consumer of the Resolved Execution Set **[repo]**

`/bin/grep -c execution_set` → **0** in `run_scorer_meta_optimizer.sh`,
`generate_scorer_jobs.py`, `scorer_trial_worker.py`, `scripts_coordinator.py`.

WATCHER freezes the set before the step (`agents/watcher_agent.py:1484`), and then step 2
resolves the fleet three more times, its own way:

| surface | anchor | address set |
|---|---|---|
| `scripts_coordinator` node loader | `scripts_coordinator.py:431`, `:444` | `distributed_config.json` — `.120/.154/.162`, 25 GPUs (localhost `gpu_count: 1`) |
| result collection | `run_scorer_meta_optimizer.sh:284-285` → `MultiGPUCoordinator('ml_coordinator_config.json')` | **`ml_coordinator_config.json`** — `.120/.154/.162`, 26 GPUs (localhost `gpu_count: 2`) |
| code push | `run_scorer_meta_optimizer.sh:257` | **three hardcoded IPs in the script body** |

`ml_coordinator_config.json` is a **tracked** file (`git ls-files`) that no fleet mechanism in
Beta's six-mechanism table names, and it disagrees with `distributed_config.json` on the
localhost GPU count. The hardcoded IPs at `:257` are a straight no-hardcoding violation; they
also fail silently (`|| echo "    FAILED on $node"`, no `exit`), so a stale-code push is a
warning line, not a stop.

---

### F10 — The completeness check on collected results cannot fire **[repo]**

`run_scorer_meta_optimizer.sh:285` — `coord.collect_scorer_results(1)`. The literal `1` is
passed regardless of `$TRIALS`.

`coordinator.py:2584-2588` uses that argument **only** for the shortfall warning:

```
elif len(all_results) < total_trials:
    self.logger.warning(f"Only found {len(all_results)}/{total_trials} results")
```

With `total_trials = 1`, any run that collects one or more results is silent, and the summary
line reads *"Found N / 1 results"*. A 500-trial run that returns 3 results reports no problem.
This is a VIR-1-class surface: **a check that is not checking, presenting as a pass.** Chapter
§12.4 documents the correct call, `collect_scorer_results(total_trials)` — the chapter is right
and the shell is wrong.

---

### F11 — The `--legacy-scoring` branch aborts at exit 127 **[live, execution-proven]**

`run_scorer_meta_optimizer.sh:236-237` — the `--sample-size $SAMPLE_SIZE` line has **no
trailing `\`**, so `--legacy-scoring` on `:237` is parsed as a separate command.

Reproduced in isolation with the same construct:

```
fake_generate --trials 10 --sample-size 450
shellprobe.sh: line 7: --legacy-scoring: command not found
exit=127
```

With `set -e` (`:165`, and `set -euo pipefail` at `:9`) the script dies before jobs are
generated. **Reachable from WATCHER:** `agent_manifests/scorer_meta.json` declares
`"flag_args": ["legacy_scoring"]`, and `agents/watcher_agent.py:1732-1735` emits
`--legacy-scoring` whenever `final_params['legacy_scoring']` is truthy. It is not in
`default_params`, so it fires only if something injects it.

---

### F12 — Three disagreeing declarations of Step 2's inputs **[repo]**

| declaration | inputs | anchor |
|---|---|---|
| manifest `required_inputs` (what WATCHER's freshness gate checks) | `bidirectional_survivors_binary.npz`, `train_history.json`, `holdout_history.json` | `agent_manifests/scorer_meta.json`; consumed at `agents/watcher_agent.py:530` |
| manifest `inputs` + `default_params.survivors` | **`bidirectional_survivors.json`**, `optimal_window_config.json`, `lottery_file` | same file |
| `PreflightChecker.STEP_INPUTS[2]` | **`bidirectional_survivors.json`**, `optimal_window_config.json` | `preflight_check.py:142` |

Two of the three name `bidirectional_survivors.json`, which is **SUPERSEDED as survivor data**
and is 2 bytes (`[]`) on this box. The manifest also still declares `"pipeline_step": 2` while
its own `description` says *"Step 2.5"* — the numbering conflict is inside a single file.

---

### F13 — The "validated operating point" sample size cannot reach the worker via WATCHER **[repo]**

Chapter §9.4 states `sample_size=450` as the validated optimum, and
`agent_manifests/scorer_meta.json` sets `default_params.sample_size: 450`.

But the manifest declares `"arg_style": "positional"` with `"positional_args": ["trials"]`, and
`agents/watcher_agent.py:1720-1728` emits **only** the declared positional args. `sample_size`
is never passed. The shell then uses its own default, `SAMPLE_SIZE=5000`
(`run_scorer_meta_optimizer.sh:171`), which is what reaches
`generate_scorer_jobs.py --sample-size` and thence the worker.

So the manifest's 450 is a configured value with no path to execution — same shape as the
`recommended_window_size` case (§2.7 instance 5b): the value is plausible, the wiring is
absent. **Reported, not fixed.**

---

### F14 — Step 2 loads survivors with format fallback enabled **[repo]**

`scorer_trial_worker.py:166` — `load_survivors(survivors_file, return_format='array')`.
`allow_fallback` defaults to `True` (`utils/survivor_loader.py:106`), and the NPZ→JSON fallback
is at `:161-166`. Step 2 does not pass `allow_fallback=False`.

Bounded Phase 6 Wall A exercised a **Step-2 load without fallback**. The production Step-2
worker is not that call. The fallback target is `bidirectional_survivors_binary.json` (via
`Path.with_suffix`), which does not exist on this box, so today it raises rather than
substituting — but the guard is a coincidence of the filesystem, not of the call.

---

### F15 — Step 2 has no dataset-provenance binding at all **[repo]**

- The worker reads **only** the NPZ (`scorer_trial_worker.py:151-283`). It never opens
  `daily3.json`, never opens `daily3_current.json`, never resolves a pointer manifest.
- `train_history.json` / `holdout_history.json` are **Step 1 side effects written to hardcoded
  repo-root paths** — `window_optimizer.py:1005-1022`. They carry no manifest identity, no
  digest, no lineage. Step 2 hard-gates on their **existence** (`run_scorer_meta_optimizer.sh:213-218`)
  and then ignores their **contents** (`scorer_trial_worker.py:527-529`, *"args 2+3 accepted for
  WATCHER/shell compat but ignored in load_data"*).
- WATCHER's P0.5 resolver is a deliberate no-op for step 2's declared inputs:
  `p05_resolve_dataset_path` returns `p` unchanged unless the basename is the legacy alias
  (`agents/watcher_agent.py:495-508`), and none of step 2's three inputs is.

The NPZ does carry lineage — through the D3.5 generation sidecar — but **Step 2 never reads the
sidecar.** Its entire binding to a dataset is the mtime comparison in `check_output_freshness`
(`agents/watcher_agent.py:571-577`).

---

### F16 — In the current topology WATCHER cannot start step 2 **[repo+live]**

`agents/watcher_agent.py:1488` runs `_run_preflight_check(step)` before dispatch;
`:1398-1407` treats any failure containing `ssh` / `unreachable` as a **hard block**.
`preflight_check.py:207-213` adds `"SSH unreachable: ..."` for any node in
`distributed_config.json` that does not answer — and that file holds the **bare-metal**
addresses `.120/.154/.162`, which are down while the rigs are booted into Proxmox.

This is a boot-state consequence, not a defect in either file (the bare-metal addresses are
deliberate, per `CLAUDE.md` §3 and §4 of the project facts). It is recorded because it changes
what F1/F2 mean operationally: **on a WATCHER-driven soak in the current topology, step 2 is
blocked before it can invoke the converter.** A direct
`bash run_scorer_meta_optimizer.sh` bypasses preflight entirely and is not blocked.

---

## 2. Q1 — the numbering, resolved from source

**Answer: there are two schemes, only one of them is executable, and the mapping between them
is folklore — it is written down nowhere in the repo.**

**Which stage does WATCHER step 2 execute?** The scorer meta-optimizer, via
`STEP_SCRIPTS[2] = "run_scorer_meta_optimizer.sh"` (`agents/watcher_agent.py:390`),
`STEP_MANIFESTS[2] = "scorer_meta.json"` (`:401`), `STEP_NAMES[2] = "Scorer Meta-Optimizer"`
(`:412`). The manifest itself carries `"pipeline_step": 2`.

**Where does the bidirectional sieve execute, by step index?** **Step 1.**
`run_bidirectional_test` is defined at `window_optimizer_integration_final.py:1369` and is
driven from the Step-1 orchestration — `prng_sweep_orchestrator.py:48,71` imports and calls it,
and the Phase-6 Wall A/B gate exercises it as `WOI.run_bidirectional_test(use_range_miner=True)`
(`tests/phase6/wall_ab_gate.py:39,336`). `STEP_SCRIPTS[1] = "window_optimizer.py"`. **There is
no executable step whose script is the sieve.** The brief's premise is confirmed.

**Is the conceptual-vs-executable mapping documented anywhere?** **No — it is folklore, and
that is a finding.** No file in the repo states "conceptual Step 2 = the sieve, which executes
inside executable Step 1" or "Step 2.5 = executable Step 2". The nearest thing is
`scripts_coordinator.py:141-151`, which classifies a jobs file as the **string** `'step2.5'`
purely to pick a batching limit — a label, not a mapping.

**Correction to the brief's §0.2.** The brief states *"the README uses conceptual stages where
sieve = 2 and scorer = 2.5."* **The live README does not.** `README.md:14` reads
`| Step 2 | scorer meta-optimizer | Distributed scoring config optimization |` — the
**executable** numbering, matching `STEP_MANIFESTS` — and the README never lists the sieve as
a numbered step at all (`/bin/grep -in sieve README.md` → one hit, line 5, prose). So the
two-scheme conflict is narrower than stated: it is **the chapter titles and the project-facts
pipeline listing** on one side, and **README + `STEP_MANIFESTS` + manifests + preflight** on
the other. (README does drift from `STEP_NAMES` at steps 4/5 — README calls 4 "anti-overfit
training" and 5 the meta-optimizer; `STEP_NAMES` has them the other way round. Separate, minor.)

**Does Chapter 3's own text conflate the two?** Yes, repeatedly and internally:

- Title and §1.3 header: *"Scorer Meta-Optimizer (Step 2.5)"*, while its manifest says step 2.
- §15 is headed **"Chapter 13: Scorer Trial Worker"**, and the document ends *"End of Chapter
  13: Scorer Trial Worker"* — in a file named `CHAPTER_3_...`.
- "Next Chapter" announces **Chapter 14: Feature Importance**, which is neither Chapter 4 nor
  the live Chapter 14 (`docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md`).
- §6.1 labels consumers "Step 3 SurvivorScorer" and "Step 5 anti_overfit" — those are
  executable indices, used in the same table as the 2.5 conceptual label.
- Two sections are both numbered **§4.2**.
- §9.4 is appended **after** "End of Chapter", out of numeric order.

---

## 3. Q2 — is Chapter 3's described execution path still live?

**Answer: the pull architecture is live; the orchestrator the chapter draws is deprecated;
PWC-era framing survives in the prose but the dispatch path itself is not PWC.**

Producer → artifact → consumer, traced:

| component | producer | status |
|---|---|---|
| `run_scorer_meta_optimizer.sh` | `STEP_SCRIPTS[2]`, `agents/watcher_agent.py:390`; also `chapter_13_triggers.py:646` | **LIVE — reachable** |
| `generate_scorer_jobs.py` | `run_scorer_meta_optimizer.sh:229,239` | **LIVE** |
| `scorer_trial_worker.py` | `generate_scorer_jobs.py:124` → job spec → `scripts_coordinator.py` | **LIVE** |
| `scripts_coordinator.py` | `run_scorer_meta_optimizer.sh:266` | **LIVE** |
| `coordinator.collect_scorer_results` | `run_scorer_meta_optimizer.sh:281-285` | **LIVE (transport only)** |
| **`run_scorer_meta_optimizer.py`** | **nothing** | **DEPRECATED — self-declared** |

`run_scorer_meta_optimizer.py:1-8` is a bare deprecation banner: *"DEPRECATED - January 9,
2026. This script assumes /shared/ml/ NFS mount which does not exist. Use
run_scorer_meta_optimizer.sh instead (PULL architecture)."* **Chapter §1.3's pipeline diagram
puts this file at the top of the stage**, with `coordinator.py (distributes to 26 GPUs)`
beneath it. Both boxes are wrong: the `.py` is dead, and dispatch is `scripts_coordinator.py`.
The chapter contradicts itself — §12.1 correctly states the `coordinator.py` →
`scripts_coordinator.py` change, then §1.3 keeps the pre-change diagram.

**On the PWC/SSH-vintage question.** The stage is SSH-dispatched, but it is **not** the
retired PWC/ZMQ path: `scripts_coordinator.py` is a script-job dispatcher over plain SSH per
`distributed_config.json`, with the explicit failure taxonomy the chapter's §12.1 describes
(`FailureMode.MISSING/EMPTY/INVALID_JSON/TIMEOUT/SSH_ERROR/EXECUTION_ERROR`,
`scripts_coordinator.py:190-197`). `persistent_worker_coordinator.py` is not in this path.
So the headline is **not** "Chapter 3 documents a dead path" — it is F3/F4 (right path, wrong
version) and F1/F2 (right path, prohibited action).

**One genuinely dead sub-path within a live file:** the `--params-file` branch
(`scorer_trial_worker.py:571-593`) that chapter §6.3, §11.2 and §12.2 present as the current
convention. `generate_scorer_jobs.py:130` emits the params as **inline JSON at argv position
5**, consumed by the backward-compatibility branch at `scorer_trial_worker.py:604-606`. Nothing
in the repo generates a `--params-file` invocation. The chapter's "v3.2 fix" is real code with
no producer.

---

## 4. Q3 — THE SEAM: what does this stage consume, and did the producer change under it?

**Answer: the 22-array contract fits the consumer exactly. The seam is sound. The path to the
artifact is what is broken.**

**What Step 2 actually reads.** Seven arrays, all by name, from the NPZ only
(`scorer_trial_worker.py:160-262`): `seeds`, `forward_matches`, `reverse_matches`,
`bidirectional_count`, `intersection_ratio`, `trial_number`, `skip_mode`. Plus one optional
side file, `optimal_window_config.json`, read for `prng_type`/`mod` (`:268-283`) — and **those
two values are then never used** by v4.3 (they are returned by `load_data` and passed to
`run_trial` as `prng_type=`/`mod=`, which ignores them).

**All seven are in the frozen 22-array contract** — verified live from
`utils.canonical_arrays.CANONICAL_ARRAY_CONTRACT`: `seeds` (1), `forward_matches` (2),
`reverse_matches` (3), `trial_number` (6), `bidirectional_count` (12), `intersection_ratio`
(14), `skip_mode` (21). **RANGE-MINER's certified artifact satisfies every column Step 2
needs, with nothing missing and nothing extra required.** The interface contract holds: this
consumer cannot tell which engine produced the bundle.

**Is what it reads still produced?** Yes — as `.s172_accumulator/current/bidirectional_survivors_binary.npz`,
surfaced at repo root through the D3.5 finalizer-owned symlink (`utils/run_finalizer.py:1400-1404`).
In the certified steady state, `run_scorer_meta_optimizer.sh:140-150` finds the symlink
resolvable and prints *"NPZ file up-to-date, skipping conversion"* — the correct outcome, and
it reads the certified generation.

**The failure is entirely in the fallback.** With the symlink absent (today) or dangling, the
script converts from `bidirectional_survivors.json` — **SUPERSEDED as survivor data**, present
here as 2 bytes — and lands a regular file on the finalizer's path (F1, F2). So the answer to
"did the producer change under it" is: **the producer changed correctly, and the stage kept a
pre-miner recovery path that now actively damages the new producer's output.**

**Dataset resolution.** Neither. Step 2 opens no lottery file of any kind — see F15. The
`daily3.json`-vs-`daily3_current.json` question does not arise for this stage, and it has no
dataset binding to lose.

**The falsy-zero idiom.** Not present in this stage — the stage reads no draw records at all.
Checked one level up, at the producer of the split it *would* have used:
`window_optimizer.py:1005-1011` uses **direct indexing**, `full_history = [d["draw"] for d in
lottery_data]`, guarded by `"draw" in lottery_data[0]`. That is not the `entry.get("draw") or …`
form, so the **22 legitimate zero-draw records are not dropped there.** `/bin/grep` for
`get("draw") or` across the Step-1/Step-2 chain: no hits.

---

## 5. Q4 — `forward_matches` / `reverse_matches`

**Answer: Step 2 sits after the columns and before the Step-3 merge. It hard-requires them,
then ignores them, and passes nothing per-seed downstream. It is neither where they are lost
nor where they could be saved.**

**Does it see them?** Yes, and it will not start without them.
`scorer_trial_worker.py:186-197`: if either key is absent, `load_data` raises
`RuntimeError('NPZ missing forward_matches or reverse_matches. Re-run
convert_survivors_to_binary.py with NPZ v3.0+ format.')` — a hard fail, plus a second guard at
`:250-251`.

**Does it use them?** **No.** They are sliced into `fwd_arr` / `rev_arr` at `:362-363` /
`:373-374` and never referenced again. They were live in v4.2 — the `bal = 1 - |mean(fwd) -
mean(rev)|` balance term documented at `:318` — and v4.3 removed the term without removing the
requirement. So the stage refuses to run without the only independent per-seed sieve signal and
then computes an objective that is independent of it.

**Does it pass them through or drop them?** Neither, in any way that reaches Step 3. Step 2's
only surviving output is `optimal_scorer_config.json` — eleven scalar hyperparameters
(`run_scorer_meta_optimizer.sh:306-307`). The worker does emit a full-length per-seed array,
but it is a **0/1 membership mask, not a score** (`scorer_trial_worker.py:459-464`,
`full[sample_idx] = mask.astype(np.float32)`), written into each trial JSON as `scores`
(`:512-513`). That array is then:

1. read by `coordinator._read_local_scorer_results` (`coordinator.py:2601-2606`), which
   **deletes the file after reading**, and
2. discarded — the collection block uses only `trial_id` and `accuracy`
   (`run_scorer_meta_optimizer.sh:291-294`).

So a per-seed array is computed, serialized, transported across the cluster, and thrown away
every trial.

**Does Chapter 3 claim to use them?** Only obliquely and now incorrectly: §1.2's feature table
says *"NPZ-Based Objective | v4.2: bidirectional_count from NPZ"*, and §7.2's code block uses
`bidirectional_count` and `intersection_ratio`. Neither is in the live objective (F3). The
chapter never mentions `forward_matches`/`reverse_matches` by name.

**Relation to the Step-3 merge-list gap.** Step 2 is upstream of it and does not touch it. The
columns' absence from the Step-3 merge list is unaffected by anything here, and **nothing was
changed** — it needs the governed schema decision Beta called for.

---

## 6. Q5 — claim-by-claim verification

Every anchor below was obtained in this audit against `575378e`.

### 6.1 Header and identity

| # | claim | verdict | anchor |
|---|---|---|---|
| 1 | Version 4.2 | **FALSE** | live objective is v4.3, `scorer_trial_worker.py:413-443` |
| 2 | File `scorer_trial_worker.py` | **ACCURATE** | file exists, is dispatched |
| 3 | Lines ~640 | **ACCURATE** | `wc -l` → **641** |
| 4 | "Execute single scorer meta-optimization trial **on remote GPU**" | **FALSE** | no GPU work remains; see #21–#26 |

### 6.2 §1–§3 overview and pull architecture

| # | claim | verdict | anchor |
|---|---|---|---|
| 5 | Receives params via JSON file or CLI args | **STALE** | live path is inline JSON at argv 5, `generate_scorer_jobs.py:130` → `scorer_trial_worker.py:604-606`; `--params-file` (`:571`) has no producer |
| 6 | "Scores survivors using residue/temporal parameters" | **STALE** | residue params only, as a k-of-3 **mask**, `:380-391`; temporal params unused (F6) |
| 7 | "Evaluates on holdout set" | **FALSE** | holdout ignored, `:527-529`, `:151-155` |
| 8 | Writes result JSON locally for coordinator to pull | **ACCURATE** | `:495-518`; `coordinator.py:2591-2610` |
| 9 | §1.3 diagram: `run_scorer_meta_optimizer.py` orchestrates | **FALSE** | that file is DEPRECATED, `run_scorer_meta_optimizer.py:1-8` |
| 10 | §1.3 diagram: `coordinator.py` distributes to 26 GPUs | **FALSE** | dispatch is `scripts_coordinator.py`, `run_scorer_meta_optimizer.sh:266`; the chapter's own §12.1 says so |
| 11 | Result path `scorer_trial_results/trial_*.json` | **ACCURATE** | `:499-501` |
| 12 | §3.2 example filenames `trial_0.json … trial_99.json` | **FALSE** | 4-digit zero-padded, `trial_{trial_id:04d}.json`, `:501` (§12.2 has this right) |
| 13 | Coordinator deletes remote file after successful pull | **ACCURATE** | local: `coordinator.py:2606`; remote: `_pull_remote_scorer_results`, `coordinator.py:2612+` |
| 14 | "26 GPUs" | **UNVERIFIABLE as written** | `distributed_config.json` → 1+8+8+8 = **25**; `ml_coordinator_config.json` → 2+8+8+8 = **26**. Both are live and both are used by this stage (F9) |

### 6.3 §4 version history

| # | claim | verdict | anchor |
|---|---|---|---|
| 15 | v4.2 is CURRENT | **FALSE** | v4.3 is current (F3); no v4.3 entry exists in §4.1 |
| 16 | v3.6 = "neg-MSE → Spearman; random.seed(42) → per-trial seed" | **FALSE** | the live docstring attributes both to **v3.5** (`:15-20`); **v3.6** is *"BUG FIX: NPZ branch never read prng_type from config"* (`:12-13`) |
| 17 | v3.5 = "degenerate guard; draw-history dependency still present", January 2026 | **FALSE** | live v3.5 is dated 2026-02-20 and is the Spearman/seed fix (`:15`) |
| 18 | v3.4 critical fix — holdout uses SAME sampled seeds | **ACCURATE as history, MOOT as behaviour** | `:31-35` records it; v4.0 removed holdout entirely (`:5-10`). The chapter itself says so at §7.3 |
| 19 | v4.2 "REMOVE: ReinforcementEngine, SurvivorScorer" implied by §4.2's redesign text | **ACCURATE** | `:8`, `:103`; neither is imported |
| 20 | v3.3 "3.8x performance improvement" via GPU batch | **UNVERIFIABLE** | benchmark artifact not in repo; the code it measured is gone |

### 6.4 §8–§9 GPU scoring and memory batching — the chapter's largest error

`scorer_trial_worker.py` defines **exactly seven functions**: `_best_effort_gpu_cleanup` (125),
`load_data` (151), `run_trial` (286), `_reject` (400, nested), `_log_trial_metrics` (469),
`save_local_result` (495), `main` (522).

| # | claim | verdict | anchor |
|---|---|---|---|
| 21 | `score_survivors_gpu()` (§8.1, §14.2) | **FALSE — does not exist** | function list above |
| 22 | `score_with_adaptive_batching()` (§9.2, §14.2) | **FALSE — does not exist** | " |
| 23 | `compute_metrics()`, `sample_survivors()`, `load_json()`, `setup_gpu()`, `parse_arguments()`, `load_params_from_file()`, `save_result()` (§14.1–14.3) | **FALSE — none exists** | " (nearest live name: `save_local_result`) |
| 24 | Scoring via `SurvivorScorer.extract_ml_features_batch()` | **FALSE for this stage** | `SurvivorScorer` is not imported (`:8`, `:103`). The **method** does exist, at `survivor_scorer.py:547`, and is Step 3's |
| 25 | §9.3 `PYTORCH_HIP_ALLOC_CONF` and `set_per_process_memory_fraction(0.8)` at file top | **FALSE — neither string is in the file** | `/bin/grep` → 0 hits |
| 26 | PyTorch/CuPy batch processing; OOM retry loop | **FALSE** | the only torch/cupy use is best-effort cleanup, `:130-146`. `run_trial` is pure NumPy on CPU |

The stage still *binds* a GPU (`:75-98`, `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`) and
still batches jobs per GPU through `scripts_coordinator`. It just never computes on one.

### 6.5 §6 trial parameters

Live space: `generate_scorer_jobs.py:52-70`.

| parameter | chapter range | live range | verdict |
|---|---|---|---|
| `residue_mod_1` | 5–20 | 5–20 | **ACCURATE** |
| `residue_mod_2` | 50–150 | 50–150 | **ACCURATE** |
| `residue_mod_3` | 500–1500 | 500–1500 | **ACCURATE** |
| `max_offset` | 1–15 | 1–15 | **ACCURATE** |
| `temporal_window_size` | 50–200 | **50–100** | **FALSE** (`:58`) |
| `temporal_num_windows` | 1–10 | **3–10** | **FALSE** (`:59`) |
| `min_confidence_threshold` | 0.05–0.50 | **0.05–0.25** | **FALSE** (`:60`) |
| `hidden_layers` | categorical | `128_64 / 256_128_64 / 512_256_128` | **ACCURATE** |
| `dropout` | 0.1–0.5 | 0.1–0.5 | **ACCURATE** |
| `learning_rate` | 1e-4–1e-2 | 1e-4–1e-2 log | **ACCURATE** |
| `batch_size` | [32,64,128,**256**] | **[32,64,128]** | **FALSE** (`:66`) |

Not in the chapter's table but in the live space: `sample_size` (`:69`) and
`optuna_trial_number` (`:119`) — both are injected into the job params and both are read by the
worker (`:348-349`).

| # | claim | verdict | anchor |
|---|---|---|---|
| 27 | §6.1 note: rm1/rm2/rm3 + max_offset + tw_size + tw_windows + min_conf feed Step 3 SurvivorScorer | **ACCURATE** | `survivor_scorer.py:97-111`, `:396-397`, `:670-672`, `:704` |
| 28 | §6.1 note: hidden_layers + dropout + lr + batch_size feed Step 5 anti_overfit | **FALSE** | F7 |
| 29 | §6.1 note: "Nothing to remove from the search space" | **FALSE** | F6, F7 |
| 30 | §6.2 `scorer_jobs.json` format (`study_name`/`study_db`/`trials[]`) | **FALSE** | live output is a flat **list** of jobs with `job_id`/`analysis_type`/`script`/`args`/`expected_output`/`timeout`, `generate_scorer_jobs.py:121-146` |
| 31 | §6.3 "OLD v3.1 long SSH / NEW v3.2 params-file" | **STALE** | neither form is generated; see Q2 |

### 6.6 §7 training/holdout split and objective

| # | claim | verdict | anchor |
|---|---|---|---|
| 32 | Step 1 outputs `bidirectional_survivors.json` | **SUPERSEDED** | 2-byte `[]` on box; live survivor artifact is the D3.5 generation + symlinks, `utils/run_finalizer.py:129-136` |
| 33 | Step 1 outputs `train_history.json` (80%) / `holdout_history.json` (20%) | **ACCURATE** | `window_optimizer.py:1013-1022`; live files are 14454 / 3614 records = exactly 80/20 of 18068 |
| 34 | §7.2 objective code block | **FALSE** | F3 |
| 35 | §7.2 "`bidirectional_selectivity` dropped (98.8% at floor)" | **ACCURATE as v4.2 history** | `:302-304`; the array remains in the 22-array contract (position 16) |
| 36 | §7.3 metrics `accuracy` / `bid_median` / `ir_median` / `n_survivors` | **FALSE** | live metrics are `enrich`, `p_global`, `p_sub`, `coverage`, `size_penalty`, `objective`, `subset_n`, `keep`, `reason` — `:479-491` |
| 37 | §7.3 note "v3.4's holdout split is no longer relevant" | **ACCURATE** | `:527-529` |

### 6.7 §10–§13 serialization, CLI, integration

| # | claim | verdict | anchor |
|---|---|---|---|
| 38 | §10.1 result JSON with `metrics{mean_train_score, generalization_gap, top_k_holdout_score, …}`, `runtime_seconds`, `gpu_id` | **FALSE** | live keys are exactly `trial_id`, `params`, `accuracy`, `status`, `error`, `hostname`, `timestamp`, `scores` — `:504-516` |
| 39 | §10.2 `save_result()` writing `trial_{trial_id}.json` | **FALSE** | `save_local_result()`, `trial_{trial_id:04d}.json`, `:495-518` |
| 40 | §10.3 `run_trial_safe()` wrapper | **FALSE — does not exist** | error handling is inline in `main`, `:619-637` |
| 41 | §11.1 flags `--residue-mod-1/2/3`, `--max-offset`, `--temporal-window-size` | **FALSE — none is parsed** | live flags: `--params-file`, `--params-json`, `--optuna-study-name`, `--optuna-study-db`, `--gpu-id`, `--use-legacy-scoring`, `:539-612` |
| 42 | §11.1 undocumented live flags | **omission** | `--params-json` (`:583`), `--use-legacy-scoring` (`:601`), `--gpu-id` (`:76-82`) |
| 43 | §12.1 Step 2.5 uses `scripts_coordinator.py`, not `coordinator.py` | **ACCURATE** | `run_scorer_meta_optimizer.sh:266` |
| 44 | §12.1 explicit failure modes MISSING/EMPTY/TIMEOUT | **ACCURATE** | `scripts_coordinator.py:190-197` |
| 45 | §12.2 `generate_scorer_jobs()` code block | **FALSE** | live jobs carry `analysis_type`, inline-JSON params (not `--params-file`), and a **ramdisk-prefixed** path `/dev/shm/prng/step2/…` — `generate_scorer_jobs.py:33-40`, `:121-136`. The chapter never mentions the ramdisk at all |
| 46 | §12.3 dispatch command | **ACCURATE** | matches `run_scorer_meta_optimizer.sh:266-269` verbatim |
| 47 | §12.4 `collect_scorer_results(total_trials)` | **ACCURATE in the chapter, WRONG in the shell** | F10 |
| 48 | §12.5 JSON 258 MB / 4.2 s vs NPZ 0.6 MB / 0.05 s | **UNVERIFIABLE** | measurement artifact not in repo; current `bidirectional_survivors.json` is 2 bytes |
| 49 | §12.5 conversion command `convert_survivors_to_binary.py …` | **ACCURATE but PROHIBITED** | F1. Documenting it is fine; the shell **auto-invoking** it is the problem |
| 50 | §13.1 "Holdout score = 0 → verify holdout_history.json exists" | **STALE** | no holdout score exists |
| 51 | §13.2 debug command with `--residue-mod-*` | **FALSE** | those flags do not exist (#41) |

### 6.8 §9.4 resource scaling (appended 2026-01-18)

| # | claim | verdict | anchor |
|---|---|---|---|
| 52 | `sample_size=450` is the validated operating point | **STALE / unreachable** | F13 — the shell forces 5000 (`:171`), the generator defaults 25000 (`:85`) |
| 53 | `max_concurrent_script_jobs=12` | **FALSE** | live per-node value is **8** (`distributed_config.json`, `ml_coordinator_config.json`), and the step-2.5 batch cap is **6** (`scripts_coordinator.py:105`, `:177`) |
| 54 | throughput table (15.41 trials/min at 450, etc.) | **UNVERIFIABLE** | benchmark artifacts not in repo; measured against the v3.x GPU scorer that no longer exists |
| 55 | "TPE ranking is preserved across sample sizes" | **UNVERIFIABLE, and now doubly so** | the objective it was measured against was replaced (F3), and 7 of 11 dimensions carry no signal (F6) |

**Tally.** 55 substantive claims assessed: **17 ACCURATE**, **9 STALE**, **24 FALSE**,
**5 UNVERIFIABLE**. Chapter 1's audit found 9 of 41; this is 17 of 55 — the same order of
magnitude. The brief's base rate held.

---

## 7. What a future session should not have to rediscover

1. **Executable step numbering is the only numbering with a source of truth**
   (`agents/watcher_agent.py:387-417`). "Step 2.5" exists only in prose and in one
   `scripts_coordinator` batching label. The mapping between the two schemes is written down
   **nowhere** — do not go looking for it again.
2. **The bidirectional sieve is inside executable Step 1**
   (`window_optimizer_integration_final.py:1369`). There is no executable step that *is* the
   sieve.
3. **There are two watcher files.** `agents/watcher_agent.py` (147 KB, has step 0/TRSE) is
   live; root `watcher_agent.py` (72 KB, no step 0) is a stale duplicate. Both are tracked.
   Anchor against `agents/`.
4. **`run_scorer_meta_optimizer.py` is dead; `.sh` is live.** The chapter's diagram says
   otherwise.
5. **Step 2 does no GPU compute.** It binds a GPU and batches per-GPU, but `run_trial` is pure
   NumPy. Do not chase a missing CuPy/torch path.
6. **Step 2 reads no dataset.** No `daily3.json`, no `daily3_current.json`, no sidecar. The
   two history files it validates for existence are ignored contents-wise.
7. **The seven arrays Step 2 needs are all in the 22-array contract.** Verified live from
   `CANONICAL_ARRAY_CONTRACT`. The miner seam is sound; do not re-derive it.
8. **`skip_mode == 1` means `'variable'`** (`utils/prng_encoding.py:37`), i.e. hybrid. Any
   reasoning about the v4.3 "rare island" depends on this.
9. **`ml_coordinator_config.json` is a fleet definition nobody counts.** It is tracked, live,
   used by Step 2's collection path, and absent from Beta's six-mechanism table.
10. **`git log -S` on the behaviour, not the file.** `scorer_trial_worker.py`'s last commit is
    a `chore` about moving docs; that commit is where the objective changed (F4). File-level
    log reading would have missed it.
11. **The 2-byte `bidirectional_survivors.json` at repo root is `[]`**, and is still named as a
    required input by two of the three places that declare Step 2's inputs (F12).

---

## 8. Explicitly not done

- **No fix, anywhere.** No file in the repository was modified except the creation of this
  report. `git status` before this report: one untracked file
  (`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_3_ALIGNMENT_AUDIT.md`).
- **`convert_survivors_to_binary.py` was NOT invoked.** It was read only
  (`:20-71`, `:104`, `:178-247`). D3.0-B is open; TB prohibits running it.
- **WATCHER was not run. The pipeline was not run.** `run_scorer_meta_optimizer.sh` was not
  executed. The only execution in this audit was (a) a direct call to `run_trial` with
  synthetic in-memory arrays and (b) a five-line shell snippet in the scratchpad reproducing a
  line-continuation defect — neither touches the repo, the cluster, or any artifact.
- **`java_lcg_cpu` non-zero-skip mismatch:** no contact point found in Chapter 3 or the Step-2
  chain. `survivor_scorer.py` is a Step-3 consumer of Step 2's *config output* only.
- **Chapters 5, 6, 8, 13:** out of scope, not examined.
- **The `forward_matches`/`reverse_matches` Step-3 merge-list gap:** reported (Q4), untouched.

---

## 9. Phase-7 step-range confinement — the mitigation for F1/F2

*Added after the audit, at owner request. Phase 7 is a **Step-1 soak**: 50 window-optimizer
trials driving the miner across 25 GPUs. Steps 2–6 are not in scope. This section records how
the confinement works, how to confirm it, and where it does not reach. **Verified in this
audit; no file was modified.***

### 9.1 The mandatory range and why the default is the hazard

```
--start-step 1 --end-step 1
```

| flag | default | anchor |
|---|---|---|
| `--start-step` | `1` | `agents/watcher_agent.py:3043-3048` |
| `--end-step` | **`6`** | `agents/watcher_agent.py:3049-3054` |

Both are passed straight through at `:3378` →
`run_pipeline(start_step=1, end_step=6, …)` (`:2328`). **Omitting `--end-step` runs the whole
pipeline**, which is exactly how step 2 gets reached. The bound itself is real and is the loop
guard:

```python
# agents/watcher_agent.py:2365
while self._pipeline_running and self.current_step <= end_step:
```

With `end_step = 1`, `run_step(2)` (`:2385`) is never called, so
`run_scorer_meta_optimizer.sh:87` (the TB-prohibited converter, **F1**) and `:97` (the `mv`
over the D3.5 finalizer-owned symlink, **F2** → `utils/run_finalizer.py:1406`
`PublicationError`) are both unreachable. Confined to step 1, the root symlink is created by
`finalize_run` during **step 1's own publication** — the correct owner.

### 9.2 The line to confirm, emitted before trial 1

`agents/watcher_agent.py:2336`, logged once at entry to `run_pipeline`:

```
Starting pipeline from step 1 to 1
```

### 9.3 ⚠ "Triggering Step 2" is NOT "Step 2 executing"

**A correctly confined run will contain the string "Step 2" in its log.** Do not treat that as
the abort signal.

`_handle_proceed` advances the counter and logs the advance **before** the loop guard
re-tests:

```python
# agents/watcher_agent.py:1143-1146
next_step = step + 1
self.current_step = next_step
logger.info(f"Triggering Step {next_step}: {STEP_NAMES.get(next_step, 'Unknown')}")
```

So on a clean `--end-step 1` run the log ends with
`Triggering Step 2: Scorer Meta-Optimizer`, the `while` at `:2365` then evaluates `2 <= 1` as
false, and the loop exits **without ever calling `run_step(2)`**. That line means step 1
succeeded and the pipeline stopped. It is the expected terminal line.

**The real abort signal** is the loop-body banner at `:2381`, emitted only inside the loop and
only four lines before `run_step(step)` at `:2385`:

```
STEP 2: Scorer Meta-Optimizer (run #1)
```

| log line | anchor | meaning |
|---|---|---|
| `Starting pipeline from step 1 to 1` | `:2336` | confinement confirmed, before trial 1 |
| `Triggering Step 2: Scorer Meta-Optimizer` | `:1146` | **benign** — step 1 done, loop about to exit |
| `STEP 2: Scorer Meta-Optimizer (run #N)` | `:2381` | **ABORT NOW** — step 2 is executing |

Grep for the banner form, not for `Step 2`:

```bash
/bin/grep -nE '^.*STEP 2: ' <run-log>     # any hit → abort
```

### 9.4 The launch must redirect stderr, or there is no run log to confirm against

`agents/watcher_agent.py:309-313` configures logging with `logging.basicConfig(...)` and
**installs no `FileHandler`** (`/bin/grep -n FileHandler agents/watcher_agent.py` → the only
hits are `:653` and `:3258`, both for the separate `watcher_decisions.jsonl` decision log, not
for `logger`). Every line quoted in §9.2 and §9.3 therefore goes to **stderr**.

A launch of the form `nohup python3 … > soak.log &` captures stdout only and produces a log
containing **none** of them — the range confirmation would be impossible to perform, and the
`STEP 2:` abort signal would be invisible. **`2>&1` is required**, e.g.
`nohup … --start-step 1 --end-step 1 > soak.log 2>&1 &`.

### 9.5 What the step range does NOT close: the Chapter-13 standalone door

`chapter_13_triggers.py:630` `execute_standalone(steps)` carries its **own** `STEP_SCRIPTS`
map, independent of `agents/watcher_agent.py:387`, with the same step-2 target:

```python
# chapter_13_triggers.py:646
2: "run_scorer_meta_optimizer.sh",
```

and `TriggerAction.FULL_PIPELINE` resolves to `[1, 2, 3, 4, 5, 6]` (`:454-455`). **`--end-step`
does not bound this path** — it is a different entry point entirely, so it would invoke
`run_scorer_meta_optimizer.sh` with F1 and F2 fully live.

**It is human-gated, not autonomous.** The only caller is the module's own `__main__` at
`:932` (`success = manager.execute_standalone(args.steps)`), reached by a deliberate
`python3 chapter_13_triggers.py --approve` / `--steps`. There is no daemon, no timer and no
enabled unit behind it (§9.6). **Operational rule: do not approve a Chapter-13 retrain while
the soak is running.**

**Not a risk — checked and cleared:** `agent_manifests/window_optimizer.json` declares
`"follow_up_agents": ["scorer_meta_agent"]`, but that field is **inert**. It is only ever
reported as a result field (`agents/agent_core.py:331`) and `next()` merely returns the name
(`agents/manifest/agent_manifest.py:388`); nothing in the tree dispatches on it. Step
advancement is owned solely by the `run_pipeline` loop, which is bounded.

### 9.6 Box state at the time of this audit **[live]**

- `ps -eo pid,etime,cmd` → **no** `watcher_agent`, `chapter_13`, `window_optimizer` or
  `range_miner` process running.
- `systemctl list-unit-files --state=enabled` → **no** unit matching `prng|watcher|daily3|chapter`.
- `systemctl list-timers --all` → **no** timer matching `prng|watcher|daily3`.

No soak is in flight and no background invoker can reach step 2 on its own.

### 9.7 Launch-time checklist

1. Command carries **both** `--start-step 1 --end-step 1` — `--end-step` is the one that
   matters; its default is `6`.
2. Command redirects **`2>&1`** into the run log (§9.4), and uses `nohup`, never `tmux`.
3. Before trial 1, confirm the log contains `Starting pipeline from step 1 to 1` (§9.2).
   Confirm it **in the log**, not only in the command line.
4. Watch for `STEP 2: ` (§9.3). Any hit → abort. Ignore `Triggering Step 2` — that is the
   normal terminal line.
5. Do not approve a Chapter-13 retrain for the duration (§9.5).
6. **Launch is Michael-initiated.** Per `CLAUDE.md` §1.3, no agent launches the pipeline.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every verdict in §1 and §6 carries a `file:line` anchor obtained in this
  audit at `575378e`. Two findings carry live execution proof: **F5** (`UnboundLocalError`
  raised by the live `run_trial`) and **F11** (exit 127 from the reproduced shell construct).
  The 22-array contract was enumerated by importing
  `utils.canonical_arrays.CANONICAL_ARRAY_CONTRACT` under the venv, not read from a document.
- **clean control:** `NOT_APPLICABLE` — read-only audit, no detector under validation.
- **fault-injection control:** `NOT_APPLICABLE` — same reason. (F5's Case B is a *contrast*
  case showing the sibling branch returns normally; it is not a fault-injection control for a
  detector, and is not claimed as one.)
- **completion sentinel:** `PASS`. Q1–Q5 answered. Claims 14, 20, 48, 54, 55 are individually
  marked `UNVERIFIABLE`; no question is `INCOMPLETE`.
- **unavailable-observer behaviour:** where a measurement artifact was absent (§12.5, §9.4
  benchmarks), the verdict is `UNVERIFIABLE`, never `ACCURATE`. Silence was not read as a pass.
- **audit claim scope:** **repo-scoped** unless marked `[live]`. Live-host claims: absence of
  `.s172_accumulator/` and of the root `bidirectional_survivors_binary.npz`; the 2-byte
  `bidirectional_survivors.json`; the 14454/3614 history record counts; both node-config
  contents; the F5 and F11 executions; `wc -l` counts.
- **searched surfaces:** tracked repo at `575378e`; the working tree (including untracked and
  gitignored files); `git log`/`git show`/`git log -S` history; `git check-ignore` on every
  config and artifact named; the live VM101 filesystem at repo root, `agent_manifests/`,
  `.s172_accumulator/`, `/dev/shm/prng`; live Python import of `utils.canonical_arrays` and
  `utils.prng_encoding`; live execution of `scorer_trial_worker.run_trial`. **For §9
  additionally:** the live `argparse` definitions and `run_pipeline` loop body in
  `agents/watcher_agent.py`, the `logging` configuration and a `FileHandler` sweep of that
  file, `chapter_13_triggers.py`'s independent `STEP_SCRIPTS` map and its sole caller, a
  `follow_up_agents` dispatch search across the tree, and live host state via
  `ps -eo pid,etime,cmd`, `systemctl list-unit-files --state=enabled` and
  `systemctl list-timers --all`.
- **unavailable surfaces:** the three rigs (booted into Proxmox; the bare-metal addresses in
  both node configs are unreachable, so no on-rig verification of deployed copies of
  `scorer_trial_worker.py` was possible — the `run_scorer_meta_optimizer.sh:257-261` push means
  rig copies may differ from VM101's and this was **not** checked); host systemd/cron units;
  `/dev/shm/prng` (does not exist, so the ramdisk path in every generated job is unverified
  against a populated ramdisk); the Optuna study DBs under `optuna_studies/` (not inspected);
  pre-repository archives on ser8; the S107/S108 benchmark artifacts behind §9.4 and §12.5.
- **gitignore check performed:** `.gitignore:41` is `*.json`.
  `bidirectional_survivors.json`, `train_history.json`, `holdout_history.json` and
  `scorer_jobs.json` are **ignored**; `ml_coordinator_config.json`, `optimal_scorer_config.json`
  and six of seven `agent_manifests/*.json` are **tracked** (only
  `agent_manifests/definitions.json` is ignored). No absence claim in this report rests on a
  repo-scoped search of a `.json` path.

---

**Report file:** `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` — **untracked. It must be committed before
the Phase-7 soak launches**, or item 5's clean-tree preflight will reject the tree at
finalization, hours in.
