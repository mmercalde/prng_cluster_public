# PHASE_6_P0_SCOPING_v1.md — where does a published dataset live, and what breaks when it moves?

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_SCOPING.md` (REV1).
**Base:** VM 101, `~/distributed_prng_analysis`, `main` @ **`43f6857`**, venv `~/venvs/torch`.
**Date:** 2026-07-31.

**READ-ONLY.** Nothing was created, moved, copied, published, or modified. No directory was
made, no version file, no pointer manifest. `daily3.json`, `.gitignore`, every config,
manifest and source file are byte-unchanged. No commit, no push, no WATCHER run.

**The rigs are powered off.** No SSH was attempted. All rig-side statements in §3 are
**repository-derived expectations, UNVERIFIED against a live rig** (VIR-5/VIR-6).

---

## 0. The falsifiable question, answered up front

> If the authoritative dataset moves to a published, versioned location, which consumers
> break, and what is the minimum change that keeps them working?

**Answer.** Under option (a) — move to `~/datasets/daily3/` — **17 call sites in 13 files
break**, of which **4 break silently** and **1 of the silent breaks is cross-machine and
cannot be tested tonight**. Under option (b) — publish in place — **zero consumers change**,
because the pointer's resolved target keeps the pathname every consumer already uses.

**(b) is both smaller and safer, and it is not close.** The decisive fact is not the change
count; it is that option (a) has a failure mode option (b) does not have at all: if the
authority moves to `~/datasets/daily3/` while a stale `daily3.json` remains at the repo root,
every hardcoded consumer keeps reading the stale file forever, with no error, no digest
mismatch, and no gate anywhere in the system that would notice. See §4.

---

## 1. Path-resolution inventory

The question here is **who resolves the path**, not who reads the content — that was
`docs/DAILY3_CONSUMER_CONTRACT_v1.md`. Every anchor below was read this session at
`43f6857`.

### 1.A — Fully path-parameterised (VIR-2 clean control: these are **safe**, no code change needed)

These consumers obtain the path from a caller and never invent one. A published path reaches
them by changing the value passed in, not by editing code.

| # | Consumer | How the path is obtained | `file:line` | Verdict |
|---|---|---|---|---|
| 1 | `window_optimizer.py` (Step 1 entry) | CLI arg, **`required=True`** — no default at all | `window_optimizer.py:1258` | **safe** |
| 2 | `WindowOptimizer.__init__` | constructor param `dataset_path` | `window_optimizer.py:576-578` | **safe** |
| 3 | Step-1 → integration handoff | `dataset_path=lottery_file` passed through | `window_optimizer.py:791`, `:1149` | **safe** |
| 4 | `window_optimizer_integration_final` miner path | function param `dataset_path` | `window_optimizer_integration_final.py:1136`, `:1214`, `:1219` | **safe** |
| 5 | `_miner_residues_for_config` | function param | `window_optimizer_integration_final.py:269`, `:291` | **safe** |
| 6 | `_get_residues_for_config` (PWC/ZMQ) | function param | `window_optimizer_integration_final.py:245`, `:251-258` | **safe** |
| 7 | `run_trial_miner` | keyword param `dataset_path: str` | `miner/range_miner_coordinator.py:4165`, stored `:4277` | **safe** |
| 8 | `_assignment_payload` | param; emitted as payload key `"dataset"` | `miner/range_miner_coordinator.py:3327`, `:3402`, `:3421` | **safe** |
| 9 | `serve_range` | reads `context["dataset_path"]` | `miner/range_miner_coordinator.py:3496` | **safe** |
| 10 | `compute_dataset_sha256` | param | `miner/range_miner_coordinator.py:73-81` | **safe** |
| 11 | **`load_residue_window`** — the D6 shared residue authority | param `path`; `open(path)` verbatim, no normalisation, no default | `miner/range_miner_worker.py:538`, `:565` | **safe** |
| 12 | `ResidueResolver.resolve` | reads `dataset_reference` → `dataset_path` → `dataset` from the assignment payload; **fails closed** with `ResidueResolutionError` if absent | `miner/range_miner_worker.py:613-616`, `:626-631` | **safe** |
| 13 | `_sha256_file` | param | `miner/range_miner_worker.py:530-535` | **safe** |
| 14 | `sieve_gpu_worker` job loader | `job.get('dataset_path') or job.get('target_file')` | `sieve_gpu_worker.py:151`, `:161` | **safe** (path from job) |
| 15 | `sieve_filter` job loader | same shape | `sieve_filter.py:523`, `:538` | **safe** |
| 16 | `load_draws_from_daily3` | param `path` | `sieve_filter.py:174` | **safe** |
| 17 | `backtest_pools.py` | CLI `--dataset`, **`required=True`** | `backtest_pools.py:166`, `:27-29` | **safe** |
| 18 | `evaluate_pools.py` | CLI `--truth`, **`required=True`** | `evaluate_pools.py:56` | **safe** |
| 19 | `full_scoring_worker.load_lottery_history` (Step 3) | param `history_file`; CLI `--train-history` **`required=True`** | `full_scoring_worker.py:201`, `:524`, `:588` | **safe**, and see note below |
| 20 | `prediction_generator.py` (Step 6) | CLI `--lottery-history`; read via `args.lottery_history` | `prediction_generator.py:972`, `:1052` | **safe** (value supplied by manifest, §1.B #4) |
| 21 | `miner/range_miner_worker.py` **daemon CLI** | **has no dataset argument at all** — `--host`, `--port`, `--gpu-id`, `--miner-output-dir`, seed caps only | `miner/range_miner_worker.py:1390-1411` | **safe**; path is 100 % coordinator-supplied |
| 22 | `digit_sequential_sieve.py` | CLI `--draws-file`; fallback is `__file__`-independent but **`--base-dir` defaults to `os.path.expanduser("~/distributed_prng_analysis")`** — absolute, not CWD | `digit_sequential_sieve.py:198-200`, `:261`, `:264-265` | **safe from CWD drift**; see D5 note |

**Step 3 does not read `daily3.json` at all.** Its manifest `required_inputs` are
`bidirectional_survivors_binary.npz`, `optimal_scorer_config.json`, `train_history.json`,
`holdout_history.json` (`agent_manifests/full_scoring.json:6-11`, defaults `:117-122`;
`run_step3_full_scoring.sh:39-40`, `:67-68`). Those are Step-1 derivatives. **Step 3 is
outside the blast radius of any dataset relocation** — a useful constraint.

`distributed_config.json` contains **no dataset path key whatsoever** — verified by reading
the full file. Nodes carry `script_path`, `python_env`, `ramdisk_path` and GPU facts only.
Relocating the dataset requires no change to it.

### 1.B — Path supplied by a config/manifest default (one-line repoint, no code change)

These are the four places a literal `daily3.json` enters the **live pipeline**. All four are
data, not code.

| # | Where | Key | `file:line` |
|---|---|---|---|
| 1 | Step-1 manifest `default_params` | `"lottery_file": "daily3.json"` | `agent_manifests/window_optimizer.json:241` |
| 2 | Step-1 manifest `required_inputs` (preflight only) | `"daily3.json"` | `agent_manifests/window_optimizer.json:6-7` |
| 3 | Step-0 manifest `default_params` + `required_inputs` | `"lottery_data": "daily3.json"`, `["daily3.json"]` | `agent_manifests/trse.json:15`, `:6` |
| 4 | Step-6 manifest `default_params` | `"lottery_history": "daily3.json"` | `agent_manifests/prediction.json:180` |

WATCHER merges these at `agents/watcher_agent.py:1379-1385` and converts them to CLI args at
`:1479-1499` via each manifest's `args_map` (`agent_manifests/window_optimizer.json:24-25`;
`agents/step_runner/command_builder.py:88-101`). Also declarative-only:
`agents/pipeline/pipeline_step_context.py:69` (`required_inputs=["daily3.json"]`,
not enforced) and `optimal_window_config.json` `agent_metadata.inputs[0].file`
(a record of the last run, not an input).

### 1.C — Hardcoded literals (code change required to repoint)

| # | Consumer | Literal | `file:line` | On the certifying path? |
|---|---|---|---|---|
| 1 | `dataset_split.py` | `SOURCE = Path("daily3.json")` — no argparse, no override | `dataset_split.py:36`, guard `:38-40` | no (manual tool; produces the split files) |
| 2 | `system_core.py` | `self.data_file = "daily3.json"` | `system_core.py:43` | no |
| 3 | `modules/window_optimizer.py` | `self.coordinator.current_target_file or 'daily3.json'` | `modules/window_optimizer.py:54`, `:110` | no |
| 4 | `modules/window_optimizer.py` | `coordinator.current_target_file = 'daily3.json'` | `modules/window_optimizer.py:305` | no |
| 5 | `integration/sieve_integration.py` | `'dataset': 'daily3.json'` (self-test block) | `integration/sieve_integration.py:442` | no |
| 6 | `persistent_worker_coordinator.py` | `dataset_path = "daily3.json"` (smoke-test `__main__`) | `persistent_worker_coordinator.py:1933` | no |
| 7 | `trse_step0.py` (Step 0) | `--lottery-data` **`default="daily3.json"`** | `trse_step0.py:811` | Step 0 is advisory / `skip_on_fail` |
| 8 | `analyze_my_lottery_data.py` | `filename = 'daily3.json'` (overridable by `sys.argv[1]`) | `analyze_my_lottery_data.py:111-113` | no |

**Finding.** *No hardcoded literal sits on the certifying Step-1 → miner → NPZ path.* Every
link in that chain from `--lottery-file` down to `open(path)` in `load_residue_window` is a
parameter. The literals live in manual tools, self-test blocks, and the legacy `modules/`
optimizer. That is the single most important structural fact in this report, and it is what
makes option (b) in §4 nearly free and option (a) merely expensive rather than impossible.

---

## 2. Relative-path hazards — the silent-break list

**This is the highest-consequence section.** A relative path is resolved against the process's
CWD. Every entry below resolves correctly today *only* because the process happens to start in
`/home/michael/distributed_prng_analysis`. Nothing enforces that.

### 2.1 The structural defect: WATCHER checks one path and uses another

| | |
|---|---|
| **What is checked** | `agents/watcher_agent.py:489` — `required_inputs = [resolve_repo_path(p) for p in required_inputs]`, where `resolve_repo_path` (`:455-459`) joins against `REPO_ROOT`, derived from `__file__` (`:429-430`, explicitly commented *"not os.getcwd()"*). So preflight checks **`<REPO_ROOT>/daily3.json`**. |
| **What is used** | `agents/watcher_agent.py:1948-1957` — `subprocess.Popen(cmd, …)` with **no `cwd=` argument**. The child inherits WATCHER's CWD. The child receives the bare string `daily3.json` (`:1495-1499`). So Step 1 opens **`<CWD>/daily3.json`**. |

`REPO_ROOT` and `CWD` are two different resolution bases, and the gate uses the one the work
does not. Launch WATCHER from any directory other than the repo root and preflight passes
against the real dataset while Step 1 resolves somewhere else. If that somewhere-else has no
`daily3.json`, the failure is loud. **If it has one, the run is silently certified against the
wrong dataset.** This is latent today and becomes materially more likely the moment a second
copy of the file exists anywhere — which is exactly what publication creates.

Note the same asymmetry applies to `agents/step_runner/`, which defaults `work_dir` to
`Path.cwd()` in six places (`step_executor.py:61`, `:298`, `:504`;
`output_validator.py:38`, `:74`, `:103`) and passes it as `cwd=` at `step_executor.py:167`,
`:233`, `:376`. That path *is* CWD-anchored end to end, which is self-consistent — but it is a
different anchoring than `watcher_agent.py` uses for the same manifests.

### 2.2 CWD-relative resolvers — ranked by consequence

| Rank | Site | `file:line` | Break mode |
|---|---|---|---|
| **1** | **Rig worker resolves the coordinator's path string against its own CWD** | payload built `miner/range_miner_coordinator.py:3421`; consumed `miner/range_miner_worker.py:614-616`; opened `miner/range_miner_worker.py:565` | **Cross-machine.** See §3. Digest-guarded, so *wrong content* fails closed; *absent file* does not (see §3.3). |
| **2** | Step 0/1/6 via WATCHER — bare `daily3.json` on the child's command line | `agent_manifests/window_optimizer.json:241`; `trse.json:15`; `prediction.json:180`; dispatched `agents/watcher_agent.py:1948` | Silent wrong-dataset if a same-named file exists in the launch CWD (§2.1) |
| **3** | `dataset_split.py:36` `SOURCE = Path("daily3.json")` | `dataset_split.py:36` | Has an explicit `.exists()` guard at `:38-40` that prints *"Run from ~/distributed_prng_analysis/"* — so **absence is loud**. A *different* `daily3.json` in CWD is silent, and it **overwrites `daily3_midday.json`/`daily3_evening.json` in that CWD** (`:75-76`) |
| **4** | `trse_step0.py:811` `--lottery-data default="daily3.json"` | `trse_step0.py:811`, loader `:769-776` | `FileNotFoundError` — **loud**. Step 0 is `skip_on_fail: true` (`agent_manifests/trse.json:12`), so a loud failure here is *swallowed by policy* and Step 1 proceeds with default bounds |
| **5** | `system_core.py:43`, `modules/window_optimizer.py:54,110,305`, `integration/sieve_integration.py:442`, `persistent_worker_coordinator.py:1933` | as listed | Not on the certifying path; loud `FileNotFoundError` when reached |

### 2.3 Consumers that resolve relative to `__file__` or an absolute base (**not** CWD-hazardous)

Recorded under VIR-2 as the checked-and-clean set:

- `agents/watcher_agent.py:429-430` — `REPO_ROOT` from `__file__`. Correct anchoring; the
  defect in §2.1 is that it is not applied to the *dispatched* value.
- `agents/watcher_agent.py:474` — manifests loaded from `REPO_ROOT/agent_manifests/`.
- `window_optimizer_integration_final.py:112` — `repo_root or os.path.dirname(os.path.abspath(__file__))`
  for the git provenance probe.
- `digit_sequential_sieve.py:189` — `--base-dir` defaults to
  `os.path.expanduser("~/distributed_prng_analysis")`, i.e. **absolute**. Its
  `data/daily3.json` default (`:264-265`) is therefore a wrong-path defect (D5 of the
  consumer contract — `data/` does not exist) but **not** a CWD-drift defect.

### 2.4 The one silent break with no detector anywhere

`evaluate_pools.py:17-26` returns `None` when a `(date, session)` is not found, and the
caller cannot distinguish *"no draw yet"* from *"wrong dataset"* (consumer contract §3.A #13).
Point it at a truncated or older published version and it reports misses, not errors. It is
`--truth`-parameterised (safe from *drift*), but it is the consumer least able to tell you it
was given the wrong file.

---

## 3. Rig-side expectation — repository-derived, **UNVERIFIED** (VIR-5 / VIR-6)

**The rigs were not contacted. Nothing below was observed on a rig this session.**

### 3.1 What the repository says the rigs read

1. The miner worker daemon **takes no dataset argument**. Its full argparse surface is
   `--host`, `--port`, `--gpu-id`, `--device-index`, `--miner-output-dir`,
   `--heartbeat-interval` and four seed caps (`miner/range_miner_worker.py:1390-1411`).
2. The path arrives **only** in the stripe assignment payload, as the key `"dataset"`
   (`miner/range_miner_coordinator.py:3421`), whose value is the coordinator's own
   `dataset_path` string, threaded unchanged from Step 1's `--lottery-file`
   (`:4165` → `:4277` → `:3496` → `:3327`).
3. The worker reads it at `miner/range_miner_worker.py:613-616` and passes it straight to
   `open(path)` (`:565`) and `hashlib` (`:530-535`). **No normalisation, no `expanduser`, no
   repo-root join, no absolute-path requirement.**
4. Therefore: **if the coordinator sends `"daily3.json"`, each rig resolves it against that
   worker process's own CWD.** The repository-derived expectation is
   `/home/michael/distributed_prng_analysis/daily3.json`, because `distributed_config.json`
   declares `"script_path": "/home/michael/distributed_prng_analysis"` for every node
   (`distributed_config.json:8`, `:17`, `:28`, `:39`) and that is the conventional working
   directory. **Nothing in the repository proves the daemon is started with that CWD** — no
   launcher script, no systemd unit, and no deploy step for `range_miner_worker.py` exists in
   the tree (searched `*.sh`, `*.service`; only documentation matches).
5. `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md:6-10` records that Phase 6.0 found exactly
   this gap: the clone brought code but not the git-ignored dataset, and `daily3.json` was
   hand-`scp`'d to CT100 `.122`.

### 3.2 What is genuinely protected

`ResidueResolver.resolve` requires `dataset_sha256` on **every** assignment and compares it to
the digest computed **on the worker's own copy**, before any cache return and before residue
loading (`miner/range_miner_worker.py:640-652`). A rig holding *different* dataset bytes
therefore fails closed with `ResidueVerificationError`, non-retryable. **A rig cannot silently
sieve a different dataset.** That is a real, working guard and it substantially de-risks the
whole relocation question.

### 3.3 What is not protected

If the file is **absent** on the rig, `_sha256_file` (`:530-532`) raises a bare
`FileNotFoundError`. That is **not** a subclass of `ResidueError`, so it does not become
`stripe_error(retryable=False)` — it escapes as an unclassified worker crash. Same class as
defect D8 of the consumer contract (`range_miner_worker.py:575`). It is loud, but it is loud
in the wrong place and at the wrong time: **during** a run rather than **before dispatch**,
which is precisely what `RUNTIME_DATASET_PROVISIONING_CONTRACT.md:36-43` forbids.

### 3.4 Consequence for the location decision

Any published location must produce a path string that is valid **on every node**, because one
string is broadcast to all of them. Two ways to satisfy that:

- keep the string relative and ensure identical CWD everywhere (today's unproven assumption); or
- make it **absolute and identical on all nodes** — which `~/datasets/daily3/` would be
  (`/home/michael/datasets/daily3/…`, same on VM 101 and each CT100, since all run as
  `michael`), and which `/home/michael/distributed_prng_analysis/daily3.json` equally would be.

Absolutising the path is an improvement **independent of which option §4 picks**, and it is
cheap: it is a value change at `agent_manifests/window_optimizer.json:241`, not a code change.

---

## 4. (a) move vs (b) publish in place

### 4.1 Change count

| | (a) move to `~/datasets/daily3/` | (b) publish in place |
|---|---|---|
| Manifest/config values to repoint | 4 (`window_optimizer.json:241`, `:6`; `trse.json:15`, `:6`; `prediction.json:180`) | **0** |
| Hardcoded literals to edit | 8 (§1.C) | **0** |
| `dataset_split.py` | must gain an argument it does not have (`:36`, no argparse) | **0** |
| Rig provisioning destination | new absolute path, untested on any rig | unchanged from the Phase 6.0 hand-provisioned location |
| Files created | version file + pointer manifest + a new directory tree outside the repo | version file + pointer manifest |
| Consumers that must be re-verified | all of §1.A + §1.B + §1.C | the alias-maintenance step only |

**(b) is smaller by roughly 17 call sites in 13 files.**

### 4.2 Which is less likely to break silently — the decisive argument

Option (a) has a failure mode option (b) does not have **at all**.

If the authority moves to `~/datasets/daily3/` and `daily3.json` is left at the repo root
(the natural, cautious thing to do — nobody deletes a 1.38 MB dataset on move day), then:

- every consumer in §1.C keeps opening the root file and never learns the authority moved;
- `dataset_split.py:36` keeps splitting the **stale** file and overwriting the split files;
- WATCHER preflight (`agents/watcher_agent.py:489`) keeps confirming a file that is no longer
  authoritative — a green check on the wrong artifact;
- the miner's `dataset_sha256` guard **does not help**, because the coordinator computes the
  expected digest from the same stale path it dispatches (`range_miner_coordinator.py:3499`
  → `:3421`). Coordinator and worker agree perfectly on the wrong file. The guard proves
  *agreement*, not *authority*;
- nothing else in the system compares the two locations. There is no cross-file digest, no
  mtime check, no record-count check (consumer contract §5.4, §8);
- and the divergence grows silently with every subsequent publication.

Deleting the root file converts all of that into loud `FileNotFoundError`s — but does so
across 17 sites simultaneously, including a rig-side path (§3) that **cannot be tested while
the rigs are off**, and one of those sites (`trse_step0.py`) sits behind `skip_on_fail: true`
and would be swallowed.

Option (b) has no such mode. The pointer's resolved target *is* the path everything already
uses. There is exactly one file at `daily3.json` and it is the current version by
construction. A consumer cannot read a stale authority because there is no second location.

### 4.3 Recommendation

**Adopt (b): publish in place.**

```
~/distributed_prng_analysis/
  daily3-<UTC>-<sha256prefix>.json   immutable version files, siblings of the current path
  daily3_current.json                pointer manifest, atomic os.replace  (see §5.2 on naming)
  daily3.json                        UNCHANGED path; the pointer's resolved target
```

`daily3.json` becomes a **finalizer-owned compatibility alias**, exactly the pattern D3.5
already uses for the accumulator root aliases (`utils/run_finalizer.py`, §2.8 of the project
facts). Publication is: write the immutable version file, `os.replace` the alias, `os.replace`
the pointer manifest. Both `open()` and `hashlib` follow symlinks, so the alias may be a
symlink or a copy without changing any consumer's behaviour — that choice can be deferred to
the P0 implementation brief.

Beta's ruling is satisfied without contradiction: the **pointer manifest** is the atomic,
machine-readable authority (not a bare symlink — Beta's words); the **alias** is a separate
compatibility surface for the 17 consumers that predate the manifest. They are different
objects with different jobs.

**Two things (b) does not fix, and should not be claimed to fix:**

1. It does not make the dataset path absolute, so §2.1 and §3.1 remain live. Absolutising is
   an independent, cheap improvement (§3.4) and should be scoped separately.
2. It does not change that publishing *into* the repo root adds files to a directory that
   already has 200+ loose scripts. That is an aesthetic cost, and it is the one real argument
   for (a). It is not worth 17 breakage sites.

---

## 5. `.gitignore` / clean-tree interaction

**A published dataset can block certification. It depends entirely on the filename, and the
rule is not the one you would guess.**

### 5.1 The wall

`utils/run_finalizer.py:1589-1596`:

```python
if not repository_tree_clean:
    raise RunParameterError(
        "repository_tree_clean is False: the working tree has uncommitted "
        "changes, so a certified generation cannot honestly claim commit …")
```

The value is produced by `window_optimizer_integration_final.py:115-118`:

```python
porcelain = subprocess.run(["git", "-C", root, "status", "--porcelain"], …).stdout
return commit, (porcelain.strip() == "")
```

`git status --porcelain` reports **untracked** files (`??`) as well as modified ones. So **any
untracked, non-ignored file anywhere in the tree makes `repository_tree_clean` False and
`finalize_run` raises before publishing anything.** The docstring at `:105-109` is explicit
that this helper *"never 'cleans up' the answer"*.

**Gate 22 is not the wall.** `tests/test_s172_phase4_coordinator.py:1607-1610` runs the same
`git status --porcelain` but filters `if ln[3:].strip().endswith(".py")` — it only sees `.py`
files. A published `.json` or a new data directory would not red gate 22. The finalizer wall
is the one that matters.

### 5.2 What the `.gitignore` actually does — tested this session with `git check-ignore`

`.gitignore:41` is `*.json`, with negations at `:42-44`. Results:

| Candidate publication filename | `git check-ignore` | Effect on certification |
|---|---|---|
| `daily3-20260731T000000Z-513648160d35.json` (repo root) | **IGNORED** (`.gitignore:41`) | safe |
| `current.json` (repo root) | **IGNORED** | safe |
| `datasets/daily3/daily3-<ts>-<sha>.json` (in-repo subdir) | **IGNORED** | safe |
| `datasets/daily3/current.json` | **IGNORED** | safe |
| `datasets/daily3/README.md` | **NOT ignored** | **blocks certification** |
| `datasets/daily3/<version>.json.sha256` | **NOT ignored** | **blocks certification** |
| `datasets/daily3/manifest.txt` | **NOT ignored** | **blocks certification** |
| `dataset_config.json` | **NOT ignored** (`.gitignore:43` `!*_config.json`) | **blocks certification** |
| `schema_publication.json` | **NOT ignored** (`.gitignore:44` `!schema_*.json`) | **blocks certification** |
| `config_dataset.json` | **IGNORED** — see §5.3 | safe |

**Three concrete traps for the P0 implementer:**

1. **Any sidecar that is not `.json` blocks certification.** A `.sha256` companion file, a
   `manifest.txt`, or a `README.md` explaining the directory each dirty the tree. The digest
   must live *inside* the JSON pointer manifest, not beside it.
2. **Do not name the pointer manifest `*_config.json` or `schema_*.json`.** Those two
   negations are live and would un-ignore it. `current.json`, `daily3_current.json`,
   `pointer.json` are all safely ignored.
3. **Publishing outside the repo (`~/datasets/`) is git-invisible and therefore always
   clean.** This is option (a)'s one genuine advantage. It is not enough to overcome §4.2,
   and it is fully obtainable under (b) by keeping every published artifact a `.json`.

### 5.3 Incidental finding — `.gitignore:42` is dead (reported, not changed)

```
!config_*.json        # Keep config JSONs (safe & important)
```

`.gitignore` does not support trailing comments on a pattern line, so the pattern is the
literal string `!config_*.json        # Keep config JSONs (safe & important)` and matches
nothing. Verified: `git check-ignore -v config_foo.json` reports `.gitignore:41:*.json`, and
`config_manifests/parameter_registry.json` is likewise ignored by `:41`. The two adjacent
negations `!*_config.json` (`:43`) and `!schema_*.json` (`:44`), which carry no trailing
comment, both work correctly.

Consequence: files the author intended to keep tracked are ignored. They are tracked today
only because they were added before the rule, or with `-f`. **Not changed** — out of scope,
and fixing it would newly un-ignore an unknown set of files and could itself dirty the tree.
Flagged for a governed decision.

### 5.4 The tree is dirty right now

`git status --porcelain` at `43f6857` returns five entries: four untracked
`CLAUDE_CODE_BRIEF_*.md` files and `tmp/`. **A certified generation attempted right now would
fail at `run_finalizer.py:1592`**, before any dataset publication is involved. This is
pre-existing session state, not caused by this scoping, and it is worth knowing before the P0
implementation brief schedules a certifying run.

---

## 6. Bootstrap readiness — is the current file publishable as-is?

Computed read-only this session by parsing the live file.

| Property | Value |
|---|---|
| Path | `/home/michael/distributed_prng_analysis/daily3.json` |
| Size | **1,380,711 bytes** |
| mtime | 2026-03-04 16:58:27 −0800 |
| **sha256** | **`513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`** |
| Records | **18,068** |
| Span | `2000-01-01 evening` → `2026-02-26 midday` |
| Sessions | evening 9,553 / midday 8,515 |

### 6.1 Conformance to `DAILY3_CONSUMER_CONTRACT_v1.md` §9 — all ten MUSTs pass

| §9 MUST | Result |
|---|---|
| 1. top level is a JSON array | ✅ `list` |
| 2. every element an object with **exactly** `{date, session, draw}` | ✅ one key-set, 18,068/18,068, no variants |
| 3. `date` = `YYYY-MM-DD`, zero-padded | ✅ 100 % regex-match, all `str` |
| 4. `session` ∈ {`midday`,`evening`} lowercase | ✅ exactly two values |
| 5. `draw` a JSON integer, 0…999, `0` emitted as `0` | ✅ all `int`, min 0, max 999 |
| 6. `(date, session)` unique | ✅ 18,068 distinct keys |
| 7. sorted ascending by raw `(date, session)` | ✅ `keys == sorted(keys)` |
| 8. historical records immutable | n/a for v1 (this *is* the baseline) |
| 9. written atomically | n/a for v1 (a copy, not a scrape) |
| 10. **no `full_state` key** | ✅ absent from all 18,068 records |
| §9.11 no `null` in any field | ✅ none |

**Nothing blocks publishing this file unchanged. It is a clean version one.**

### 6.2 The 22 zero-draw records — Alpha's reasoning **confirmed**, with a stronger basis

Alpha's position: the 22 `draw == 0` records that two loaders silently drop are a **consumer
defect, not a data defect**, and the file should be published as-is because altering it would
make generation `gen-20260730T002104136270Z` (commit `b08c2c5`) unreproducible against its own
inputs.

**Confirmed on all three legs, and the reproducibility argument turns out not to be the
strongest one available.**

**Leg 1 — the records are genuine draws, not sentinels.** `000` is a legitimate California
Daily 3 outcome. Over 18,068 draws the expected count of any single value is 18.07. Observed
count of `0` is **22**; the observed per-value counts across all 1,000 values range 4…32 with
median 18, and 847 of the 1,000 values have a count ≤ 22. The zeros are unremarkable. They are
also **spread across the whole span** — 22 indices from 254 (`2000-09-11 evening`) to 16,939
(`2024-08-11 evening`), not clustered at an import boundary the way a placeholder artifact
would be. Nothing suggests a defect in the data.

**Leg 2 — the defect is unambiguously in the consumers, and it is off the certifying path.**
Both droppers use the falsy-zero idiom `entry.get("draw") or entry.get(…)`:
`digit_sequential_sieve.py:161-162` and `coordinator.py:1881` (consumer contract D1). Neither
is on the Step-1 → miner → NPZ certifying path. The certifying loader is
`load_residue_window`, which uses `entry.get("full_state", entry["draw"])`
(`miner/range_miner_worker.py:575`) — a plain subscript, **zero-safe**. The contract's own §9
MUST #5 already states normatively that *"`0` is valid and must be emitted as `0`"*. A
producer that dropped or rewrote these records would be violating the frozen spec in order to
accommodate two consumers that are already documented as defective.

**Leg 3 — the reproducibility argument holds, and is broader than stated.** Editing the file
changes its sha256, and `dataset_sha256` is a mandatory, verified field on every stripe
assignment (`miner/range_miner_worker.py:640-652`, TB Blocker-6 Option C). Beyond replaying
D6, the 6-P1 accumulator input wall is an **exact input-manifest digest match**, so the
certified lineage would be permanently unreconcilable with its own inputs. Correct.

**One refinement to Alpha's framing.** The reproducibility argument, taken alone, would equally
forbid publishing a file that *was* genuinely defective — it argues from the cost of change,
not from the correctness of the content. It should not be the load-bearing reason. Legs 1 and 2
are: the data is right, the two loaders are wrong. Reproducibility is then the reason not to
"fix" it *later*, once someone notices the 22 records again. Both should be recorded, in that
order.

**Recommendation.** Publish as-is. Record the 22 zero-draw records and their indices in the
publication manifest's notes so the next reader does not re-litigate this. Fix D1 in the two
defective consumers under a separate, non-certifying change — **never** by touching the data.

---

## 7. Pointer-resolution insertion points

**Nothing reads a pointer manifest today, because none exists.** Repo-wide search for
`source_location|destination_path|datasets/|dataset_root|publication_dir` returns one abstract
hit (`RUNTIME_DATASET_PROVISIONING_CONTRACT.md:27-28`), confirming the brief's premise.

Beta's requirement is that a run resolves the pointer **once at start** and every node uses
that frozen version. Tracing where that is achievable:

### 7.1 It is **one** insertion point for the certifying path — and a second for everything else

**(A) The certifying path: one point.** `dataset_path` is threaded, unbranched, from Step 1's
CLI arg to every node:

```
window_optimizer.py:1258  --lottery-file  (required)
  → :791 / :1149  dataset_path=lottery_file
    → window_optimizer_integration_final.py:1214-1219
      → range_miner_coordinator.py:4165 run_trial_miner(dataset_path=…)
        → :4277 context["dataset_path"]
          → :3496 serve_range
            → :3327/:3421 _assignment_payload → payload["dataset"]
              → range_miner_worker.py:614 → :565 open(path)
```

Resolving the pointer once at the head of that chain freezes every node, because the nodes
never resolve anything themselves (§3.1). **No rig-side insertion is needed.** This is a
genuinely favourable topology and is the reason P0 is small.

**(B) The rest of the pipeline: one point, at WATCHER's param merge.**
`agents/watcher_agent.py:1385` (`final_params = {**default_params}`) is the single place where
manifest defaults become the values dispatched to Steps 0, 1 and 6. Resolving the pointer once
there — and injecting the resolved absolute path plus the frozen digest into `final_params` —
makes all three steps of a pipeline run agree on one version. It would also close the §2.1
check-one-path/use-another defect as a side effect, since the injected value would be absolute.

### 7.2 Where the freeze is **not** currently achieved — the gap P0 must not paper over

`miner/range_miner_coordinator.py:3499` computes `dataset_sha256 = compute_dataset_sha256(dataset_path)`
inside `serve_range`. Its comment says *"coordinator-computed ONCE and reused for every
assign"* — which is true **per trial**, not per run. `serve_range` is invoked once per
Optuna trial, so a Step-1 run of N trials re-hashes the file N times. A scrape landing between
trials 3 and 4 would put the two halves of one study on different bytes, each internally
consistent, with no error. This is the concrete shape of the freeze-at-run-start requirement:
the digest must become an **input** to the coordinator, resolved at (A) or (B) above and
passed down, rather than re-derived at `:3499`.

### 7.3 Entry points that would each resolve independently unless given a shared resolver

Four direct-CLI consumers bind their own path and would not see a run-level freeze:
`trse_step0.py:811`, `backtest_pools.py:166`, `evaluate_pools.py:56`,
`prediction_generator.py:972`. Plus `dataset_split.py:36`, which has no argument at all.
Under option (b) these keep working against the alias, which is the pointer's resolved
target — so they are **consistent by construction** without any change. Under option (a) each
would need its own resolver call. **A further point in (b)'s favour.**

**Summary: one insertion point for the certifying run (A), one for the WATCHER-driven pipeline
(B), and — under option (b) only — zero for everything else.**

---

## 8. Minimum viable P0

Beta: P0 needs *"one valid immutable version and a pointer manifest"*, produced by a manual
bootstrap.

### 8.1 What must exist for P0

1. **The publication schema, frozen.** Version ID = UTC timestamp + content identity, e.g.
   `daily3-20260731T<hhmmssffffff>Z-<sha256[:12]>.json`. Must be documented before anything is
   written, because the first version file's name is permanent.
2. **One immutable version file** — a byte-exact copy of the current `daily3.json`
   (sha256 `513648160d35…68f6`, 18,068 records, 1,380,711 bytes), published beside it.
   **Filename must end in `.json`** so `.gitignore:41` keeps the tree clean (§5.2).
3. **One pointer manifest**, atomically replaced (`os.replace`), naming the current version
   and carrying, *inside the JSON*: version ID, `sha256`, `size_bytes`, `record_count`,
   first/last `(date, session)`, publication UTC, lineage ID, and a notes field recording the
   22 zero-draw records (§6.2). **No sidecar files** — a `.sha256` or `.txt` companion dirties
   the tree (§5.2). **Do not name it `*_config.json` or `schema_*.json`** (§5.2).
4. **The compatibility alias**: `daily3.json` stays exactly where it is, as the pointer's
   resolved target (§4.3). Nothing else in P0 touches it.
5. **The correction protocol, documented** — a correction opens a new lineage; the old lineage
   is preserved. Documentation only in P0; no enforcement code.
6. **A read-only verifier** that re-derives the version file's digest and confirms it matches
   both the manifest and the alias. This is the P0 clean control; without it P0 has published
   something nothing has ever checked.

### 8.2 Honestly deferrable to **P0.5** (fleet provisioning)

- The provisioning manifest of `RUNTIME_DATASET_PROVISIONING_CONTRACT.md:18-34`
  (`source_location`, `destination_path`, `owner`/`group`/`mode`, `verification_command`,
  `failure_behavior`) and the per-node verification loop.
- **Fail-before-dispatch enforcement** (`:36-43`) — including converting the bare
  `FileNotFoundError` at `range_miner_worker.py:530-532` into a classified `ResidueError`
  (§3.3).
- **Freeze-at-run-start wiring** — moving the digest at `range_miner_coordinator.py:3499` from
  per-trial derivation to a run-level input (§7.2).
- **Pointer resolution at WATCHER** (§7.1 B) and **absolutising the dispatched path** (§3.4,
  §2.1). These are cheap and high-value; they are deferred only because they are *changes to
  running code*, whereas P0 is *creating files*, and mixing the two makes the first
  certification after publication ambiguous.
- Anything requiring a rig. **The rigs are off; none of this is verifiable tonight** (VIR-5).

### 8.3 Honestly deferrable to **P2** (the scraper)

- Append-only production of new versions, the publication-prefix wall (a record-sequence
  check, not a byte-prefix test), `CORRECTION_REQUIRED` halt behaviour, and re-enabling
  `daily3scraper.service`.
- Regeneration of `daily3_midday.json` / `daily3_evening.json`, and any binding between them
  and the combined file (consumer contract §5.4, §9.23). **P0 publishes the combined file
  only.** The split files remain unversioned, unbound, and stale-by-default — unchanged by
  this scoping, and flagged as an open item so P0 is not mistaken for having addressed them.

### 8.4 Not P0, but should be scheduled

- **D1 falsy-zero fix** in `digit_sequential_sieve.py:161-162` and `coordinator.py:1881`
  (§6.2). Non-certifying, and must not be bundled with a data change.
- **`.gitignore:42` dead negation** (§5.3) — needs a governed decision, since fixing it
  un-ignores an unknown set of files.
- **The §2.1 check-one-path/use-another defect** in WATCHER — arguably a bug fix independent
  of Phase 6 entirely.

---

## 9. Coverage table and completion sentinel

### 9.1 Verification-integrity controls (VIR-1…6)

- **execution proof:** every consumer claim in §1, §2, §5, §7 carries a `file:line` read at
  `43f6857` this session. The §5.2 ignore table is `git check-ignore` output, not inference.
  The §6 figures are a full parse of all 18,068 records, and the sha256 was computed twice by
  independent tools (`sha256sum` and `hashlib`) with identical results.
- **clean control (VIR-2):** §1.A lists **22 consumers verified as already path-parameterised
  and safe**, and §2.3 lists **4 resolvers verified as `__file__`- or absolute-anchored and
  therefore not CWD-hazardous**. §6.1 is a ten-row all-pass conformance table. The report is
  not breakage-only.
- **fault-injection control:** **n/a — stated explicitly rather than omitted.** This is
  read-only scoping; injecting a fault would require creating or corrupting a file, which the
  brief forbids. The nearest available substitute was used instead: `git check-ignore` was run
  against **negative** cases (`README.md`, `.sha256`, `manifest.txt`, `dataset_config.json`,
  `schema_publication.json`) and correctly reported them NOT-ignored, which proves the §5.2
  probe distinguishes and is not vacuously reporting "IGNORED" for everything.
- **completion sentinel (VIR-3):** §9.3 below.
- **unavailable-observer (VIR-5):** the three rigs are powered off. No SSH was attempted. §3 is
  labelled repository-derived and UNVERIFIED throughout; it is **not** reported as clean.
- **audit claim scope (VIR-6):** §9.2 below. **The repository is not the system.**

### 9.2 Surfaces searched and unavailable

**Searched:** the repo tree at `43f6857` via `/bin/grep` (not the ugrep wrapper, which honours
`.gitignore` and would have skipped every `.json`) for `daily3|lottery_history|pa_pick3`,
`lottery_file`, `dataset_path`, `repository_tree_clean`, `porcelain`, `cwd=`, `STEP_SCRIPTS`,
`gate.?22`; `git ls-files` to separate tracked source from the ~200 archival/backup copies;
`git status --porcelain`; `git check-ignore -v` against ten candidate publication filenames;
full read of `.gitignore`, `distributed_config.json`, `optimal_window_config.json`, the three
relevant `agent_manifests/*.json`, `docs/DAILY3_CONSUMER_CONTRACT_v1.md`,
`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md`; full parse of `daily3.json`;
`ls` of `/home/michael/datasets` (**does not exist**) and of the five dataset files at the
repo root.

**Unavailable / not searched:**

- **The three rig CT100s (`.122`/`.156`/`.164`) and bare-metal `.127`** — powered off, no SSH.
  Their dataset copies, digests, worker CWDs, launch mechanism and any deployed-but-uncommitted
  files were **not observed**. All of §3 is repository-derived expectation.
- **Host surfaces outside the repo on VM 101** — systemd units, cron, `/home/michael/*.py`,
  `/home/michael/cluster_controller/`. Not re-swept this session; the consumer contract's §0
  sweep of these is cited, not reproduced.
- **The ~200 archival files** matching the dataset name (`backups/`, `*.before_*`, `*.bak`,
  `apply_s1*.py` patch scripts, `step6_restoration/`, `coordinator.py.*`). Enumerated as a set
  and deliberately excluded from §1 as non-live. If any is ever revived it must be re-checked.
- **Other git branches and deleted-file history.**
- **A whole-file semantic read of all live `.py` files.** Method was keyword + pattern
  enumeration; a consumer that builds the path from a fully computed string with no literal on
  the line would evade it.

### 9.3 Coverage table

| Brief § | Required finding | Status | Where |
|---|---|---|---|
| 2.1 | Every consumer of the dataset **path**, with `file:line`, classified parameterised vs hardcoded | **PASS** | §1.A (22 safe), §1.B (4 config-supplied), §1.C (8 hardcoded) |
| 2.2 | Relative-path hazard — the silent-break list | **PASS** | §2, incl. the structural check-one-path/use-another defect §2.1 |
| 2.3 | What the rigs actually read | **UNAVAILABLE** (repository-derived expectation reported and labelled; rigs off, no SSH) | §3 |
| 2.4 | (a) move vs (b) publish-in-place — smaller and safer | **PASS** — (b) on both counts | §4 |
| 2.5 | `.gitignore` / clean-tree / gate-22 / `repository_tree_clean` interaction | **PASS** — publication *can* block certification; filename-dependent; three concrete traps | §5 |
| 2.6 | Bootstrap: sha256, record count, schema conformance, blockers; confirm-or-challenge the 22 zero-draws | **PASS** — nothing blocks; Alpha confirmed, with a refinement to the framing | §6 |
| 2.7 | Pointer-manifest readers and insertion points — one or several | **PASS** — one for the certifying path, one for WATCHER, zero elsewhere under (b) | §7 |
| 2.8 | Minimum viable P0, with P0.5 / P2 deferrals | **PASS** | §8 |
| 4 | VIR-1…6 controls, incl. explicit fault-injection n/a | **PASS** | §9.1 |

### 9.4 Completion sentinel

```
PHASE_6_P0_SCOPING_v1 — SENTINEL

OVERALL:  PASS (with one UNAVAILABLE sub-finding)

  §2.1 path inventory ............ PASS
  §2.2 relative-path hazards ..... PASS
  §2.3 rig-side expectation ...... UNAVAILABLE  (rigs powered off; repository-derived
                                                 expectation reported, NOT verified,
                                                 NOT treated as clean — VIR-5)
  §2.4 move vs publish-in-place .. PASS  (recommend (b) publish in place)
  §2.5 gitignore / clean tree .... PASS  (publication CAN block certification;
                                          filename-dependent — see §5.2)
  §2.6 bootstrap readiness ....... PASS  (nothing blocks; publish as-is)
  §2.7 pointer insertion points .. PASS  (one certifying + one WATCHER; zero elsewhere)
  §2.8 minimum viable P0 ......... PASS

  fault-injection control ........ n/a (read-only scoping) — stated, not omitted

  MUTATIONS PERFORMED: none. No file created, moved, copied, published, modified,
  staged, committed or pushed. No directory created. No SSH. No WATCHER run.
  Working tree at end of session is byte-identical to `43f6857` plus this document.

  STOPPED AT THE GATE FOR TEAM ALPHA REVIEW.
```
