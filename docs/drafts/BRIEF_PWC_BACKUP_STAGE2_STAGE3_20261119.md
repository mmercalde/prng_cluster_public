# BRIEF — PWC full backup, Stage 2 and Stage 3 (read-only planning)

**Date:** 2026-11-19 · **Host:** VM101, as `michael` · **Model:** Opus, one session, no sub-agents.
**Code read:** main `27eb991`, extracted with `git archive` into this session's scratchpad. The code files the scope
cites have not changed between `7444aea` (the scope's base) and `27eb991`; `git diff --stat 7444aea 27eb991` touches
only `docs/`, `config_manifests/survivor_volume.cap` and one reference script. **All `file:line` anchors are main
`27eb991` unless marked "candidate".**
**Candidate:** the step-1 work at `/home/michael/tfm_worktrees/pwc-step1` (uncommitted), read only, for names and
shapes. **The `pwc-backup` worktree, where Stage 1 is being coded now, was not opened.** Stage 1 will move line
numbers in `persistent_worker_coordinator.py`. Each change below therefore names the function as well as the line.
The person doing Stage 2 re-anchors after Stage 1 lands.
**Written:** this file only. Nothing was committed or pushed. No rig, no GPU, no live ledger, no main-tree write, no
test run.

**Yardstick.** A backup engine is judged by whether Steps 2–6 can use its output exactly as they use RANGE-MINER's,
and in the end by the hit rate of pools built from it. Nothing here judges PWC by whether any seed is found. **No
hit-rate measurement of PWC-produced survivors exists.**

---

## Prior record

### Searches run (on `27eb991`, `git grep`, which does not skip `*.json`)

```
git grep -lIiE "PWC|persistent_worker" 27eb991 -- 'docs/TB_RULING_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT*.md'
for f in docs/PROJECT_FILE_CATALOG.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md \
         docs/exchange/LEDGER.md docs/exchange/DEFERRED.md; do git grep -cIiE 'PWC|persistent_worker' 27eb991 -- $f; done
git ls-tree --name-only 27eb991 docs/exchange/to_beta/ docs/exchange/from_beta/ docs/drafts/ docs/ | grep -iE "pwc|backup"
git grep -nIiE "PWC|persistent_worker" 27eb991 -- docs/exchange/LEDGER.md docs/LEADS.md
```

Hits: 30 ruling/proposal/chapter/audit files; catalog 11 lines, TODO 21, BACKLOG 0, LEADS 3 (all GCVM history, not
this subject), LEDGER 3 (X-PWC-BACKUP R1/R2), DEFERRED 0. Exchange: X-PWC-BACKUP R1/R2 and X-PWC-CACHE-RESET R1/R2.
The base document for this brief is `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md`, which already ran and recorded
the full prior-record search for this subject (its §Prior record). This brief does not repeat that table. It adds
only what is new.

### Documents that already decide or record this subject

| Document | What it settles | Label |
|---|---|---|
| `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` (all) | Items A1–A6, B1–B7, the staged plan, the shared-file table | ALREADY RECORDED. This brief is its Stage 2/3 detail |
| Owner rulings 2026-11-19 20:57 PST (relayed in the task, not yet in a repository document) | (1) lift the hybrid block once the threshold matches the miner; (2) PWC earns credit; (3) drop the seed-start refusal for PWC; (4) PWC always runs the miner's phases | **CHANGES A RULING** each. Now ruled by the owner. Cited per change below. They should be written into a repository document before Stage 2 lands [owner/lead] |
| Comparator ruling §5 (quarantine), `persistent_worker_coordinator.py:179-239` | Lifting condition was threshold **and** skip bounds | Superseded on the threshold half by owner ruling (1) |
| Owner decision 2026-11-14, R5 (`window_optimizer.py:678-712`) | PWC/ZMQ refused at a non-zero start | Superseded for PWC by owner ruling (3). ZMQ keeps it |
| Beta 2026-07-31 retirement; X-STEP1-SEARCH-CODE R1 Q1 (`window_optimizer_integration_final.py:1397-1403`) | PWC carries no credit | Superseded by owner ruling (2) |
| S147 Q0/B1 prune and S147 Q2 single strategy (`persistent_worker_coordinator.py:1779`, `:1840-1849`, `:1866-1875`) | PWC prunes on zero forward survivors, skips hybrid reverse, uses one strategy | Phases: superseded by owner ruling (4). Strategy: superseded by the owner's default rule ("PWC does what RANGE-MINER does"); see §F |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (F-1) | Miner constant phases fall back to skip `[0,16]`; owner decides after Run 4 | ALREADY RECORDED, still open. Question F4 below |
| `docs/HYBRID_SKIP_BOUND_AUDIT.md` §0; facts §2.7 #4 | Hybrid skip bounds are dead in both engines | ALREADY RECORDED, open, shared |
| `miner/range_miner_worker.py:600-605` (`GENERATOR_PHASE_V1_PIN = 0`, "Raising this is DEP-ABI-V2 and is a ruling") | Device-side generator phase is pinned to 0 | ALREADY RECORDED. Stage 2 copies the pin to PWC |
| `scripts/gate12_parity_gate.py:20-23` | Rigs are deployed by one `git clone`, then targeted `scp`; only content digests count | ALREADY RECORDED. Used in §D |

### New items this brief raises

| Item | Label |
|---|---|
| **The RANGE-MINER launch process on VM101 imports `sieve_filter.py` at module load** (`window_optimizer_integration_final.py:52-55`). Its top level sets ROCm paths and imports CuPy. It also calls `set_limit(256 MB)` on the CuPy pool of that process (`sieve_filter.py:24-70`). The scope's "not imported by `miner/`" is true for the `miner/` package and false for the launch process | **NEW.** It changes how §C must prove a `sieve_filter.py` edit safe |
| Three miner/shared suites pin PWC behaviour that the rulings change. They are `tests/test_s172_phase4_coordinator.py:1927`, `tests/test_s172_miner_seed_start.py:888-889` and `tests/test_step1_search_record_stage1.py:888-896`, `:1538`, `:1705-1706`. Also `tests/test_s172_phase5_d3_25_candidate_ingress.py:380` and `tests/test_s172_threshold_propagation.py:115`, `:720-930` | **NEW** as a consequence. Editing them is edits to the suites used as the proof; §C says how |
| PWC injects **three** HSA variables in four places, and the miner sets **none** | The HSA override is ALREADY RECORDED (Stage A 6g). The other two variables and the probe path are **NEW** |
| The TCP worker echoes the hybrid threshold rounded to 4 places (`sieve_filter.py:599-602`) | NEW, by reading. It affects how the threshold gate compares |
| The manifest's default route is PWC: `"use_persistent_workers": true`, `"use_range_miner": false` (`agent_manifests/window_optimizer.json:262`, `:270`) | NEW as a juxtaposition. How Run 4 overrides it is UNVERIFIED (scope "Not verified" already lists it) |

### Not searched

The rigs (off; deployed bytes unavailable), Proxmox, live ledgers and staging, ser8 archives, the private repository,
the project PDFs and claude.ai project, `~/dashboard_work`, private owner chat. The `pwc-backup` worktree, which was
kept out of scope on purpose.

---

## A. What the four rulings mean in code (one paragraph each)

1. **"Once the threshold is fixed (same as range miner) we lift the block."** The miner gives the hybrid kernel the
   trial's directional threshold and checks that the value it asked for is the value used
   (`miner/range_miner_coordinator.py:9303-9310`, `:9356-9361`; worker `range_miner_worker.py:1374-1380`). PWC does
   the same: it passes the threshold, checks the echo, then removes the raise. The hybrid skip bounds stay dead in both
   engines alike (shared, open §2.7 #4). No kernel changes.
2. **Credit: yes.** PWC gets a sweep-evidence function of the same shape as `_step1_miner_sweep_evidence`. A phase
   counts only when every chunk finished and the chunks tile the requested interval. Credit then flows through the
   unchanged `credit_decision`.
3. **Drop the seed-start refusal for PWC.** PWC partitions from the requested start, as the miner does from
   `base_start`. The refusal stays for ZMQ.
4. **PWC follows the miner's phases.** It runs phases 1–2 always, and 1–4 when `test_both_modes`, whatever the
   survivor counts. That is `workflow_stages_for` (`miner/range_miner_coordinator.py:12046-12062`). No prune, no Q0
   skip.

---

## B. Change lists

Each change gives the file and line, what it does in plain words, and the ruling it carries out. "Default rule" means
the owner's instruction that anything unruled copies RANGE-MINER.

### Stage 2 — PWC semantics match the miner (PWC-only files, plus `sieve_filter.py`)

Prerequisite: Stage 1 has landed (candidate S1–S8 merged over main, the work in `pwc-backup`). Stage 2 uses the
candidate's `pwc_threshold_record` / `assert_pwc_threshold_contract` (candidate `:416`, `:441`),
`build_pwc_trial_completion` (candidate `:510`) and S4's removal of the `hasattr` fallbacks.

| # | Where | What it does | Ruling |
|---|---|---|---|
| 2.1 | `persistent_worker_coordinator.py:1851-1863` and `:1878-1890` (`run_trial_persistent`, the two hybrid `run_sieve_pass` calls) | Adds `phase2_threshold=forward_threshold` and `phase2_threshold=reverse_threshold`. The 0.5 default at `:1252` no longer reaches the kernel | (1) |
| 2.2 | `run_sieve_pass` (`:1243-1256` signature; job dict `:1356-1369`) | For a hybrid pass, **refuses** if `phase2_threshold` was not given (no default). It writes the threshold into the job as both `min_match_threshold` and `phase2_threshold`, as the miner does | (1) |
| 2.3 | `run_sieve_pass` result handling; candidate S6 record | Builds a requested/payload/effective record for hybrid passes and fails the pass on any mismatch or missing echo. The TCP worker echoes `single_phase.threshold` rounded to 4 places (`sieve_filter.py:599-602`). Either the gate compares at 4 places, or 2.8 also makes the echo unrounded. **The second is closer to the miner, which compares exact values** | (1) |
| 2.4 | `:179-239` (the quarantine code, class and function) and the call at `:1279` | Removes the raise. Keeps a refusal for the **SSH** transport on hybrid passes (see 2.9), because that path cannot echo | (1) |
| 2.5 | `run_trial_persistent`: forward-zero prune `:1779-1793`; Q0 gate `:1866-1875` | Deletes both. Phases 1–2 always run; phases 3–4 run whenever `test_both_modes` and the base is not `_hybrid` (the same test as `_step1_required_phases`, `window_optimizer_integration_final.py:1388-1394`) | (4) |
| 2.6 | `:1838-1849` (the `balanced_hybrid` load) and the two `strategies=_hybrid_strategies` arguments | Passes `strategies=None`, so `run_sieve_pass` loads all five through `get_all_strategies()` (`:1281-1291`). This is the same source the miner worker uses (`range_miner_worker.py:1232`) | Default rule. Supersedes S147 Q2 (see §F, question F5) |
| 2.7 | Job dict `:1356-1369` | Sends two new fields: `window_anchor` (= `config.offset`, used for slicing the draws) and `generator_phase` (= 0, the miner's v1 pin, `range_miner_worker.py:605`). It keeps `offset` for the SSH route only | Default rule |
| 2.8 | `sieve_filter.py`, `execute_sieve_job` `:535-538`, `:595`, `:618`, `:638`, `:677`; `run_sieve` `:220`, `:311`; `run_hybrid_sieve` `:362`, `:461` | Slices the draws by `window_anchor`. Passes `generator_phase` (not the anchor) as the kernel's last argument. Refuses a job that has `window_anchor` but no `generator_phase`, or a non-zero phase. A job with neither keeps today's behaviour, so old callers are unchanged | Default rule |
| 2.9 | `sieve_filter.py:450-461` (`run_hybrid_sieve` argument list) | Builds the hybrid argument list to match the miner's builders (`range_miner_worker.py:322-331` and siblings). The forward java hybrid gets **15** arguments, not 16 (kernel `prng_registry.py:1006-1013`, which takes no offset). Reverse hybrids get 13 plus `generator_phase`. The same check applies to every family the miner supports. Families the miner does not support are refused on a counted run | Default rule |
| 2.10 | Coordinator transport selection (`__init__`/`startup`; S3 address handling from the candidate) | Refuses `pwc_transport="ssh"` on a counted run. The SSH route runs `sieve_gpu_worker.py`, which is in the miner's rig pin. It has the S153 reverse-argument mismatch and no threshold echo, and cannot be fixed without editing a miner file | Default rule (miner has one transport); Stage 0 item (a) |
| 2.11 | `:1157` (legacy TCP result parser) | Reads `skip_pattern` (what `sieve_filter.py:497-500` emits) as well as `skip_sequence`. Not one of the 22 arrays | Default rule (completeness) |
| 2.12 | Coordinator cache reset (`_reset_rig_cupy_cache`, `:153-170`) | Adds `pwc_cache_policy = per_run | keep`. The default is `per_run`, and `keep` is refused on a counted run, as the miner's v7 policy does (`9034e18`). Whether PWC also writes a cache record is question F3 | Default rule |
| 2.13 | `persistent/pwc_transport_tcp.py:393-426`, `:514-550`; `persistent/pwc_worker_service.py:357-361` | Adds a per-run launch token. The transport serves only worker ids from this run's launch set, and a worker exits on explicit rejection. This matches the miner's registration refusal and release token (`execution_set.py:930-950`; `RUN_NONCE` in `scripts/launch_fleet_manual.sh`). Level 1 (launch set) is in Stage 1 per the scope; this is level 2 | Default rule |
| 2.14 | **Only if the owner rules F2 "match the miner".** Coordinator worker env `:140`, `:141`, `:145`; probe command `:406-408`; `persistent/pwc_worker_service.py:300-301` | Stops injecting `HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA` and `HSA_ENABLE_DEBUG_TRAP` | Question F2 |
| 2.15 | **Only if the owner rules F4.** `run_trial_persistent`, constant passes (the `skip_range` argument, candidate S4 form) | Sends `[0,16]` on phases 1–2 (what the miner sieves today) instead of the sampled range | Question F4 |

Line size, from the scope: about 300–400 lines of code and 600–800 lines of tests, about 2 sessions.

### Stage 3 — shared launch plumbing (PWC-only lines inside RANGE-MINER's launch files)

| # | Where | What it does | Ruling |
|---|---|---|---|
| 3.1 | `window_optimizer_integration_final.py:1955-1963` (`_use_pw` branch) | Deletes the PWC call to `_route_start_guard` | (3) |
| 3.2 | `window_optimizer_integration_final.py:1968-1986` (`run_trial_persistent(...)` call) | Adds `seed_start=seed_start` (the candidate's S2 parameter, candidate `:2200`). Also adds `pwc_coordinator_address=` (S3) and the sizing keys from 3.8 | (3) |
| 3.3 | `window_optimizer.py:701-711` (`assert_step1_route_seed_start_supported`) | Narrows the refusal to `use_zmq_sqlite` only (and `route == "zmq"`), and rewrites the docstring. The callers at `:819-821` and `:1257-1259` are unchanged; for PWC they become no-ops | (3) |
| 3.4 | `window_optimizer_integration_final.py`, a new top-level function beside `:1397` | `_step1_pwc_sweep_evidence(pw_result)`: copies `pw_result["pwc_completion"]["swept_intervals"]` (candidate shape `{phase: [start, end]}`). It returns `None` (no credit) unless `pwc_trial_complete` is true and the completion schema matches | (2) |
| 3.5 | `window_optimizer_integration_final.py:2002-2006` | Replaces the `_step1_unaccounted_route_evidence("pwc", …)` stamp with 3.4 | (2) |
| 3.6 | `window_optimizer_integration_final.py:1987-1994` (pruned return) | After 2.5, PWC never returns `pruned`. The branch becomes a refusal ("PWC returned pruned; not permitted under the always-phases rule"), so an old worker build cannot produce an uncredited silent trial | (4), (2) |
| 3.7 | `window_optimizer_integration_final.py:1162` (`_build_test_result_from_pw`) | Before touching the accumulator, requires a validated `threshold_provenance` built from the per-pass threshold records (2.3). This mirrors the miner's wall at `:1330-1346`. PWC branch only | Default rule |
| 3.8 | `agent_manifests/window_optimizer.json` (`:37-43` arg map; `:171-200` param specs; `:262-268` defaults); `window_optimizer.py` both paths (`run_bayesian_optimization` `:729-735`, `:865-876`; `run_with_config` `:1195-1199`, `:1317-1325`; CLI wiring `:1930-1936`, `:2049-2054`); `agents/watcher_agent.py:1812-1818` (route pass-through) | New PWC-only keys: `pwc_coordinator_address` (no default), PWC chunk and pool sizing inside the recorded envelope (candidate S5), and `pwc_cache_policy`. Existing keys and values stay as they are | Default rule |
| 3.9 | `window_optimizer.py:1895-1901` (`run_start_dataset_gate(...)`); `agents/watcher_agent.py:2118-2124` (`resolve_absent_fleet_status(...)`) | Passes `require_fleet=bool(use_persistent_workers)`. Both functions already take `require_fleet` (`miner/dataset_authority.py:1066`, `:1239`). `dataset_authority.py` is not edited | Default rule (the miner fails closed on a missing provisioning manifest) |
| 3.10 | Tests listed in §E3 | Updates the assertions that pin the old PWC behaviour | Follows (2), (3), (4) |

Size, from the scope: about 150–200 lines of code and about 400 of tests, 1.5–2 sessions. This is in Beta's
before-commit class (launch path).

---

## C. Proof that RANGE-MINER is unchanged, per file it loads or launches through

Baseline for every comparison: main `27eb991` (or whatever main is when Stage 2 or 3 is cut, recorded by SHA).
"AST identity" means comparing `ast.dump(node, include_attributes=False)` of the named nodes, before and after. All of
it is CPU-only. **The suites must run one after another, not in parallel** (memory note: G-VAL-6 flakes under
concurrency).

| File (how the miner reaches it) | Stage | Proof | Proof possible? |
|---|---|---|---|
| **The ten gate12-pinned files** (`scripts/gate12_parity_gate.py:174-185`; on the rigs the miner worker executes them) | 2, 3 | `git diff --exit-code <base> -- <the ten files>` must be empty. Run `derive_worker_import_closure()` from the gate, locally (CPU, no SSH), and require the derived set to be the same as on the base, so a new import cannot pull `sieve_filter.py` or `persistent/*` into the miner worker | Yes |
| `sieve_filter.py` (**imported at module load by the miner's VM101 launch process**, `window_optimizer_integration_final.py:52-55`; never by the rig miner worker) | 2 | (a) AST identity of every **module-level statement** except the `GPUSieve` class and `execute_sieve_job`. That covers the env prelude `:24-36`, the imports, the CuPy `set_limit` block `:56-70`, `_s163_read_proc`, `_best_effort_gpu_cleanup`, `load_draws_from_daily3` (`:174-188`, used by `_get_residues_for_config` `:249-270`), `main`, and the `__main__` guard. (b) In a CPU child process: `import window_optimizer_integration_final` succeeds, and the `os.environ` diff and the `sys.modules` key set after import equal the base. (c) The gate12 closure check above, which shows the rig worker does not reach the file | Yes. (b) needs CuPy importable on VM101. Without a GPU, the `set_limit` call is inside `try` and the comparison still holds |
| `persistent_worker_coordinator.py` (imported at module load by the integration, `:57-60`; no miner path calls into it) | 2 | (a) The integration import succeeds in a child process. (b) A static check that no node inside the `_use_miner` branch of `run_bidirectional_test` references `run_trial_persistent` or the module. (c) The existing miner suites stay green (list in §E4) | Yes |
| `persistent/pwc_worker_service.py`, `persistent/pwc_transport_tcp.py`, `persistent/pwc_transport_base.py` (imported only lazily by the PWC coordinator, `:538`; `grep` finds no use under `miner/`) | 2 | `git grep -n "persistent\.\|pwc_" -- miner/` shows no import; gate12 closure unchanged | Yes |
| `window_optimizer_integration_final.py` (the miner's launch file) | 3 | (a) AST identity of: `_build_test_result_from_miner`, `_step1_miner_sweep_evidence`, `_step1_ledger_rows_via_copy`, `_miner_residues_for_config`, `_step1_required_phases`, `Step1ChunkState`, `_step1_search_preflight`, every `_flush_*` and checkpoint function, and every module-level statement. (b) In `run_bidirectional_test`: AST identity of the whole `if _use_miner:` subtree and of every statement before it; only the `if _use_pw:` subtree may differ. (c) `credit_decision` (`utils/step1_search_ledger.py:1044-1086`) byte-identical (not edited) | Yes |
| `window_optimizer.py` (miner launch file, both paths) | 3 | (a) AST identity of every function except `assert_step1_route_seed_start_supported`, `run_bayesian_optimization`, `run_with_config` and the CLI `main`. (b) For those four: the AST diff, statement by statement, may contain only the guard condition and PWC-key lines. A reviewer reads the printed diff. (c) Behaviour table for the guard: `(seed_start ∈ {0, 2³¹}) × (route ∈ {miner, pwc, zmq, legacy})`. Only ZMQ at a non-zero start raises | **Partly.** Pure AST identity is not possible for those four functions, because they carry the new lines. The proof is a diff confined to named lines, plus the behaviour table, plus `test_s172_miner_seed_start.py` green on its miner gates |
| `agents/watcher_agent.py` (WATCHER launches the miner) | 3 | (a) AST identity of every function except the one holding `:1812-1818` and the one holding `:2118-2124`. (b) The diff in those two is only `require_fleet=` and PWC keys. (c) A behaviour test: with `use_range_miner=True` and the manifest missing, it raises exactly as on the base; with neither engine flag set, the result is unchanged | **Partly**, same reason. **The change at `:2118-2124` sits inside a long WATCHER step function. No suite runs that function end to end on CPU [UNVERIFIED; not searched for]. The proof there is reading plus a unit test of the argument passed** |
| `agent_manifests/window_optimizer.json` (WATCHER reads it for every step-1 run) | 3 | Load base and new as JSON. Every base key path must have an identical value; the only differences are added keys whose names start with `pwc_`. Run the manifest/argparse parity tests that exist for S170 [suite name UNVERIFIED] | Yes |
| `utils/canonical_records.py`, `utils/canonical_arrays.py`, `utils/step1_search_ledger.py`, the finalizer (shared output door) | 2, 3 | Not edited. `git diff --exit-code` empty | Yes |
| Rig-side behaviour of the miner after a PWC redeploy (§D) | 2 | Gate12 parity on the rigs (owner-run) proves the ten files' deployed bytes did not move | **Not possible without rigs.** It waits for the owner's rig session |

**Where proof is not possible:**
1. Pure AST identity for the four `window_optimizer.py` functions and the two WATCHER functions that carry new lines.
   They are covered only by a confined diff plus behaviour tests.
2. Anything about deployed rig bytes until the owner runs gate12 and the PWC parity gate with the rigs up.
3. The **existing** side effect of `sieve_filter.py` on the miner's VM101 process (the 256 MB CuPy pool limit,
   prepended `PATH`/`LD_LIBRARY_PATH`). The proof shows it is **unchanged**, not that it is harmless.

---

## D. `sieve_filter.py` and the rigs

**Does RANGE-MINER on the rigs load it?** No, by reading:
- The miner worker imports `miner.range_miner_protocol` at module scope (`range_miner_worker.py:84`). It lazily
  imports `prng_registry` (`:634`), `hybrid_strategy` (`:1232`) and `sieve_gpu_worker` (`:1279`).
- None of those import `sieve_filter`. `git grep sieve_filter -- miner/` is empty.
- `sieve_filter.py` is not in gate12's pinned set (`scripts/gate12_parity_gate.py:174-185`).
- The rig fleet launcher (`scripts/launch_fleet_manual.sh`) runs only `range_miner_worker.py`.

The rig files were not checked (rigs off), so there may be other deployed copies [UNVERIFIED].

**But on VM101 the miner's launch process does import it** (`window_optimizer_integration_final.py:52-55`, inside
`try/except ImportError`). See §C for the proof this needs.

**What a rig redeploy involves** (the gate12 docstring `:20-23` says rigs are a `git clone` once and then targeted
`scp`):
1. Owner-initiated, with the rigs booted into Proxmox (CT100 endpoints `.122/.156/.164`, CLAUDE.md §3), after Stage 2
   has landed and been committed.
2. `scp` of only the PWC files Stage 2 changed to each CT's `script_path` from `distributed_config.json`, using
   absolute paths:
   - `sieve_filter.py`;
   - `persistent/pwc_worker_service.py` and `persistent/pwc_transport_tcp.py` (for 2.13);
   - `persistent/pwc_protocol.py` and `persistent/pwc_transport_base.py` if Stage 1 changed them.
   - **No gate12-pinned file is copied.**
3. Run the candidate's PWC parity gate (`scripts/pwc_parity_gate.py --transport tcp`). Its TCP pin includes
   `sieve_filter.py` and `persistent/*`. It must match the deployed bytes on all three rigs.
4. Run `scripts/gate12_parity_gate.py`. It must still pass with the same digests, which proves the miner's ten files
   were not touched.
5. No kernel source changes (`prng_registry.py` untouched), so no CuPy recompile is forced. The per-run cache reset
   applies anyway.

Size: one short owner session. It can be combined with Stage 4 (i).

**Can Stage 2 avoid touching `sieve_filter.py`?** Only by moving the anchor/phase split into
`persistent/pwc_worker_service.py`. That would mean either a wrapper that rewrites the job, or a windowed copy of the
dataset handed to `execute_sieve_job` with `offset=0`. This has three problems:
- `pwc_worker_service.py` is **also** rig-deployed, so the redeploy is not avoided.
- The forward java hybrid would still get its 16th argument (value 0), unless `sieve_filter.py` is edited anyway.
- The rewrite creates a second window-derivation path. That is the shape the D6 correction removed on the miner side.

A coordinator-only fix (a pre-windowed dataset per anchor) would need new dataset files on every rig. That breaks the
P0.5 dataset authority, which pins dataset digests. **So Stage 2 cannot avoid a rig redeploy in any form, and editing
`sieve_filter.py` is the smaller change.** Holding counted PWC to anchor 0 avoids every edit but cannot run Run 4's
anchor 25 (scope B3).

---

## E. Tests to add or change (all CPU-only)

### E1. Stage 2, new: `tests/test_pwc_backup_stage2.py`

Fake TCP worker, fake ssh, CuPy mocked.

- **G-THR-1 to G-THR-4.** Hybrid passes carry `phase2_threshold == directional threshold` (forward 0.71, reverse
  0.47) in the job. A missing kwarg refuses. An echo that disagrees, is missing, or is rounded beyond tolerance fails
  the pass. Constant passes are unchanged.
  - Mutants: drop the kwarg at either call site; restore the 0.5 default; skip the echo check.
- **G-QUAR.** A hybrid pass now dispatches over TCP. An SSH hybrid pass on a counted run still refuses.
- **G-PHASES.**
  - With `test_both_modes=True`, a fake worker returning 0 survivors in phase 1 still runs phases 2, 3 and 4.
  - A fake returning 0 in phase 3 still runs phase 4.
  - With `test_both_modes=False`, exactly phases 1–2 run.
  - The `pruned` key is never true.
  - Mutants: restore the prune; restore Q0.
- **G-STRAT.** The hybrid job carries exactly the five strategies of `get_all_strategies()`, in that order. Mutant:
  `balanced_hybrid` alone.
- **G-ARGS (anchor/phase split).** Capture the CuPy argument tuple from `GPUSieve.run_sieve` and `run_hybrid_sieve`
  (mock kernel). For every variant the miner supports, compare count, dtype and the last value with the miner's
  builders (`range_miner_worker.build_*`, called on CPU through `BuildContext`):
  - forward java hybrid has 15 arguments;
  - the last argument is `generator_phase == 0` at anchor 25;
  - the draw slice starts at 25.

  Also: a non-zero `generator_phase` refuses; a job without the new fields behaves byte-for-byte like the base.
  Mutants: anchor passed as phase; the 16th argument restored.
- **G-WINDOW.** `load_draws_from_daily3(path, w, sessions, anchor)` equals `load_residue_window(path, w, sessions,
  anchor)` on a fixture dataset, for both-sessions and single-session cases.
- **G-CACHE.** `per_run` resets; `keep` on a counted run refuses; `keep` on a non-counted run skips the reset (fake
  ssh).
- **G-TOKEN.**
  - A real `TcpTransport` on loopback: a listed worker with the right token is served.
  - An unlisted worker, a wrong token or a stale id is refused, and the fake worker exits on refusal.
  - Mutant: always accept.
- **G-PARSER.** A TCP result with `skip_pattern` fills the skip sequences.
- **G-HSA, G-SKIP.** Only if F2 or F4 is ruled: the env lists and the probe string contain no `HSA_*`; constant passes
  send `[0,16]`.

### E2. Stage 2, CPU proof scripts (§C)

`scripts/pwc_stage2_miner_unchanged.py` (new, read-only): the ten-file diff, the gate12 closure derivation, the
`sieve_filter.py` module-level AST comparison, and the child-process import and env/modules comparison.

### E3. Stage 3

New: `tests/test_pwc_backup_stage3.py`.
- **G-START.** A fake `run_trial_persistent` receives `seed_start=2³¹` from `run_bidirectional_test`. The guard table
  from §C holds. `run_with_config` with PWC and two iterations is no longer refused.
- **G-CREDIT.**
  - A complete four-phase PWC result gives `swept_intervals == {1..4: [s, e]}` and `credit_decision` grants.
  - Any incomplete pass, a missing completion, a wrong schema or a `pruned` result grants nothing.
  - A **zero-survivor complete** trial is credited (Beta's paired control).
  - Mutants: accounting dropped; the requested interval used in place of the swept one.
- **G-PROV.** A PWC result without validated threshold provenance is refused before the accumulator changes. Mutant:
  wall removed.
- **G-FLEET.** With `use_persistent_workers` and the provisioning manifest missing, both the CLI and the WATCHER
  callers raise; the miner and legacy routes are unchanged.
- **G-KEYS.** Manifest JSON diff contains only the added `pwc_*` keys; both `window_optimizer.py` paths and WATCHER
  pass them (S170 parity rule).

Existing tests whose PWC assertions change. Only the named assertions are edited; every miner assertion stays as it
is, shown by a diff limited to those lines:

| Test | Lines |
|---|---|
| `tests/test_s172_threshold_propagation.py` | `:115`, `:720-805`, `:919-930` (quarantine expectations become the threshold gate) |
| `tests/test_s172_phase4_coordinator.py` | `:1927` and its gate |
| `tests/test_s172_miner_seed_start.py` | `:888-889` (source mutant of the PWC guard line), plus any gate expecting PWC refusal |
| `tests/test_step1_search_record_stage1.py` | `:888-896`, `:1538`, `:1705-1706` |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | `:380` (the forward-zero shape gate; it can stay as a direct `build_trial_populations` test) |
| `test_persistent_worker_harness.py` and `test_s147_patches_harness.py` (root) | Their S147 prune and Q0 expectations, together with A4 |

### E4. Suites that must stay green, run one at a time, before and after each stage

The ten-file gate12 sha check, plus:
- `test_s172_phase4_coordinator.py`
- `test_s172_resolved_execution_set.py`
- `test_s172_phase5_d3_25_candidate_ingress.py`
- `test_s172_miner_seed_start.py`
- `test_step1_search_record_stage1.py`
- `test_step1_credit_wal.py`
- `test_s172_window_anchor_brief_i.py` and `_mutants.py`
- `test_cache_policy_v7.py`
- `test_pwc_cache_reset.py`
- the candidate's `test_pwc_backup_step1.py`

Gate 22 is red while untracked `.py` exist (memory note) and goes clean once the files are committed.

**Known red at baseline** (scope A4): G-MINER-UNCHANGED, through the stale D6 sub-harness, until that harness is
repaired. Its result must be recorded as "same red as base" and not read as a pass.

---

## F. Stage 0 items: what "match RANGE-MINER" means, or the question

| Item | What matching RANGE-MINER means | Can PWC match? | Question for the owner |
|---|---|---|---|
| (a) TCP-only for counted PWC | The miner has one transport, its own TCP protocol, with a threshold echo and a launch-set refusal. PWC's TCP route can carry the same guarantees (2.3, 2.13). The SSH route runs `sieve_gpu_worker.py` (in the miner's rig pin) with no echo and the S153 reverse-argument mismatch | Yes, by refusing SSH on counted runs (2.10) | **F1.** Is SSH allowed for non-counted diagnostic PWC runs, or removed? |
| HSA override | The miner sets no `HSA_*` variable (none in `miner/` or `scripts/launch_fleet_manual.sh`). Its lazy import of `sieve_gpu_worker` only sets them on hostnames `rig-6600*` (`sieve_gpu_worker.py:28-30`), and the CT100s report `rrig6600*` (CLAUDE.md §3). PWC injects three variables at `persistent_worker_coordinator.py:140-145`, `:406-408` and `persistent/pwc_worker_service.py:300-301` | Yes, technically (2.14) | **F2.** Drop all three to match the miner? Note: `HSA_ENABLE_DEBUG_TRAP=0` is the S165 fix for a KFD crash during concurrent reset and init. PWC's whole operating history ran with these set. Their effect on gfx1032 under the current ROCm is UNVERIFIED |
| (g) Cache record | The miner's launcher writes a per-run, per-worker fresh-cache record. The record verifier and join gate check it (`9034e18`: `scripts/cache_records_verify.py`, `scripts/gate12_cache_join_gate.py`, `gate12_launch.sh`). PWC resets the cache and prints a marker, but writes no record | Partly. PWC can write records in the same schema from its own reset. Checking them with the miner's verifier means calling a miner launch script on PWC output; that is a read-only use, but not an edit | **F3.** Should counted PWC runs write v7-schema cache records, and should `cache_records_verify.py` (unchanged) be the check? |
| F-1 order | The miner's constant phases sieve at `[0,16]`, because no `skip_range` is sent (`range_miner_worker.py:1293`). The trial's sampled `skip_min`/`skip_max` are only recorded (`range_miner_coordinator.py:2666`, `:2691`). PWC sends the sampled range | Yes (2.15), but that copies the behaviour F-1 calls open | **F4.** Should PWC copy `[0,16]` now, keep the sampled range, or wait for your F-1 decision? Until then, the two engines differ on phases 1–2 whenever the sampled range is not `[0,16]` |
| S147 Q2 strategy set | The miner always loads all five strategies | Yes (2.6) | **F5.** Ruling (4) covers phases. Confirm that the default rule also replaces S147 Q2's "balanced_hybrid for discovery" (5× the work on phases 3–4) |
| S147 prune as an optional flag | The miner has no prune flag | Yes, by deleting it (2.5) | **F6.** Keep the S147 behaviour reachable behind a non-default flag (yardstick rule point 4), or delete it outright as the miner has none? |
| Survivor-volume cap | `config_manifests/survivor_volume.cap` (50,000 per phase, `7c725c8`) is checked by a manual pre-launch script through the miner's estimator (`utils/survivor_volume_estimator.py`). Nothing on the launch path reads it | Yes, once 2.1–2.9 make PWC sieve like the miner. The same manual gate then describes PWC | None. A runbook line |
| `min_workers = 1` (owner ruling 2026-08-04) | The miner admits by the frozen execution set | Not needed for coverage: completion accounting makes a short fleet show up as no credit | None (status) |

---

## G. Not verified

- **Deployed rig bytes:** what `sieve_filter.py` and `persistent/*` each rig carries now; real TCP/ssh behaviour of
  the token, kill, reset and reconnect. Rigs off.
- **CT100 hostnames at runtime** (`rrig6600*` in CLAUDE.md §3, not checked live). They decide whether the
  hostname-gated HSA prelude in `sieve_filter.py:27-29` and `sieve_gpu_worker.py:28-30` fires.
- **Kernel arguments:** whether CuPy/HIP tolerates the 16th argument to the forward java hybrid today (harmless or
  not), and whether the miner's builders cover every variant the TCP route can be asked for. G-ARGS would show the
  second.
- **The VM101 import side effect:** whether `sieve_filter.py`'s CuPy pool `set_limit(256 MB)` at import affects the
  miner coordinator process in practice. Only "unchanged" is provable.
- **The threshold echo path:** whether the 4-place rounding of the echo would pass an exact-equality gate for every
  sampled threshold.
- **Test coverage of WATCHER:** whether any existing suite runs the WATCHER P0.5 block at `:2118-2124` on CPU. Not
  searched.
- **The S170 manifest/argparse parity test:** its name. Not searched.
- **Stage 1 as landed:** the anchors and names here come from main and the `pwc-step1` candidate. The `pwc-backup`
  worktree was not read, so Stage 1 may rename functions this brief relies on.
- **Run 4's override:** how Run 4 overrides the manifest defaults (`use_persistent_workers: true`). This is inherited
  from the scope's list.
- **The rulings' wording:** the four owner rulings exist here only as relayed text. No repository document records
  them yet.
- **Sub-agents:** none were used. Every anchor above was read in this session, except those taken from the scope and
  marked as its readings: the miner threshold checks at `range_miner_coordinator.py:9303-9310` and `:9356-9361`,
  worker `:1374-1380`, `execution_set.py:930-950`, and the TCP transport line ranges.
