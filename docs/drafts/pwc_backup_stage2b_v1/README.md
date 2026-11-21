# PWC backup — Stage 2b (round 2): PWC moves to RANGE-MINER's window-anchor / generator-phase method

**Date:** 2026-11-20 · **Host:** VM101, as `michael` · **Model:** Opus, one session, no sub-agents.
**Worktree:** `/home/michael/tfm_worktrees/pwc-offset`. It was moved from `ee39404` onto main `0caeaf9` (Stage 2 landed)
with the round-1 edits kept.
**Status:** uncommitted. Nothing was committed, pushed, relayed to Beta or deployed. No rigs, no GPU, no live ledger.

**Yardstick.** This change is judged by one thing. On the routes that can carry the split, PWC must build its sieve
input the way RANGE-MINER does:
- the trial's offset picks the draw window;
- the generator phase is 0;
- the hybrid kernels get the miner's argument lists.

Then later steps cannot tell the engines apart on that input. Nothing here measures hit rate. Nothing here judges PWC
by whether any seed is found. The PWC yardstick question, hit rate against `k/1000` on unseen draws, stays open.

---

## Prior record

### Searches run

Round 1's searches are recorded in round 1's copy of this README (`logs/stage2b_pre_rebase/stage2b_pre_rebase_newfiles.tgz`)
and in the Beta request's Prior record:
- the window-anchor documents and the miner reads;
- the mandated `docs/` scopes, plus `docs/exchange/`, for PWC × anchor/phase, `sieve_filter.py:18x`, Brief II, the
  Stage 2b names and the proof tool.

Added this round (worktree at `0caeaf9`):

```
git log --oneline ee39404..0caeaf9 ; git diff --stat ee39404 0caeaf9
git diff ee39404 0caeaf9 -- persistent/pwc_worker_service.py tests/test_pwc_backup_step1.py tests/test_pwc_stage1.py
grep -n "validate_pwc_window_offset|offset" persistent_worker_coordinator.py
grep -n "offset|G-OFFSET|M-OF" docs/drafts/pwc_backup_stage2_v1/README.md tests/test_pwc_backup_stage2.py
grep -rln "sieve_filter|execute_sieve_job|run_hybrid_sieve|GPUSieve" --include=*.py --include=*.sh .
sed -n 1752,1950p window_optimizer_integration_final.py          # the engine cascade
sed -n 170,185p miner/range_miner_worker.py                      # EXPECTED_KERNEL_ARITY
sed -n 172,186p scripts/gate12_parity_gate.py                    # the ten pinned files
```

### Documents that decide or record this subject

| Document | What it settles | Label |
|---|---|---|
| Owner ruling 2026-11-20 08:30 PST | Offset picks the draw window only; `generator_phase` is 0; PWC moves to RANGE-MINER's method | Owner ruling. In the repo as `docs/drafts/pwc_backup_stage2_v1/README.md:50`, `:196` and `persistent_worker_coordinator.py:330-343` |
| Owner answers, 2026-11-20 (this session) | (c) the old-offset refusal goes at PWC's entry points only, and `execute_sieve_job` keeps old-style jobs for the legacy `coordinator.py` and ZMQ engines; (a) the SSH route keeps Stage 2's refusal of offset ≠ 0; (b) all four no-phase forward hybrids get the miner's counts, for PWC new-key jobs only | Owner decisions. Not yet in a repository document |
| `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §2, §4.1, §4.2 | Both keys required, `offset` HARD REJECT, anchor checked and never clamped | ALREADY RECORDED. Copied |
| `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`; `docs/TODO_CONSOLIDATED.md:89` | Brief II is BLOCKED until Brief I is accepted. No document records that acceptance | **CHANGES A RULING** in sequence, **ruled by the owner 2026-11-20 17:38 PST**, verbatim: "Yes, the 830 lifts the PWC wait so that we can get all methods on the same page. Proceed". The block is lifted for PWC only |
| `docs/drafts/pwc_backup_stage2_v1/README.md:196-200`, open question 1 (`:659-663`) | Stage 2 refused offset ≠ 0 until the split landed, and asked for the worker edit | ALREADY RECORDED. This change is that split. The refusal stays on the SSH route |
| `miner/range_miner_worker.py:813-860`, `:913-925`, `:171-185`, `:570-600`; `miner/range_miner_coordinator.py:2731-2732` | The method copied: refuse `offset`, require both keys, check the anchor domain, use the per-variant argument counts | ALREADY RECORDED (read only) |
| `docs/CHAPTER_9_AUDIT.md:68-73` (B-1) | The ROCm hostname check in `sieve_filter.py:27` never matches the live `rrig6600*` hosts | ALREADY RECORDED. Not changed |

### Not searched

Rigs (off), deployed rig bytes, ser8, the private repository, claude.ai project files and PDFs, Proxmox, and the
`pwc-backup` worktree. Of that worktree, only Stage 2's runner, suite list and compare scripts under
`logs/pwc_stage2_v1/` were read, then copied here.

---

## Step 1: moving onto `0caeaf9`, and the joined worker diff

- **Saved first:**
  - `/tmp/stage2b_pre_rebase.diff` (2 files) and `/tmp/stage2b_pre_rebase_newfiles.tgz` (12 entries), as ordered;
  - the same files plus whole-file copies in `logs/stage2b_pre_rebase/`, because `/tmp` is cleared on reboot
    (CLAUDE.md rule 7).
- **The move:** `git checkout -m --detach 0caeaf9`. Three-way merge, no conflict markers.
  - `sieve_filter.py` was byte-identical to the saved copy; Stage 2 did not touch it.
  - My `_execute_job` hunk and Stage 2's `PWC_EXIT_HELLO_REJECTED = 43` hunk do not overlap. Both survive (`:44`,
    `:369`, `:563-588`).
- **The joined diff of the worker against `0caeaf9`:** `joined_pwc_worker_service_vs_0caeaf9.diff`. It now holds the
  round-2 form, below.

---

## Step 2: the four decisions, settled by "PWC does what RANGE-MINER does"

### (b) first: can RANGE-MINER's launch reach `GPUSieve.run_hybrid_sieve`? No

Proved by reading:
1. **The only callers** of `run_hybrid_sieve` are inside `execute_sieve_job` (`sieve_filter.py:652`, `:695`; repo grep;
   the other hits are stale copies and patch scripts, not imported).
2. **What the miner imports:** the miner's VM101 launch process imports from `sieve_filter` only
   `load_draws_from_daily3` (`window_optimizer_integration_final.py:52-55`). It calls it from `_get_residues_for_config`
   (`:249-255`). That function is unchanged.
3. **The miner branch never falls through:** in `run_trial`, the miner branch either raises or returns
   (`:1761-1944`: `raise ImportError`, `return TestResult` at `:1927`, `return _miner_tr` at `:1944`). The PWC (`:1950`),
   ZMQ (`:2013`) and legacy (`:2117`) branches come after it.
4. **Nothing else on the miner side reaches it:** no file under `miner/` and neither launcher (`gate12_launch.sh`,
   `scripts/launch_fleet_manual.sh`) references `sieve_filter`, `execute_sieve_job` or `run_hybrid_sieve`. The rig
   miner worker never imports `sieve_filter.py` (it is not in gate12's governed set).

So the edit was allowed and was made. Its import-time effect is proven unchanged below.

### (a) The coordinator sends `window_anchor` + `generator_phase`; Stage 2's refusal is replaced by the miner's behaviour

`persistent_worker_coordinator.py`:

| Lines | What |
|---|---|
| `:370` | `PWC_GENERATOR_PHASE_V1_PIN = 0`. The miner's pin (`range_miner_worker.py:605`) is restated, not imported, so PWC does not import `miner/` |
| `:373-385` `validate_pwc_window_anchor` | Integer check. A negative anchor is refused, and so is anything that is not a plain int (`bool`, `float`, `str`). The upper bound needs the dataset, so the worker checks it. **SSH route:** delegates to Stage 2's `validate_pwc_window_offset`, so offset ≠ 0 is still refused there (owner answer (a); `sieve_gpu_worker.py` is pinned and reads one `offset` for both jobs) |
| `:388-399` `pwc_new_key_job` | Copies the job with `offset` replaced, in its place in the key order, by `window_anchor` and `generator_phase` 0. Nothing else changes |
| `:402-417` `pwc_anchor_echo_error` | The worker must echo exactly the anchor and phase the job carried. A worker from before Stage 2b drops both keys and sieves the window at anchor 0 without saying so. This is where that run is caught instead of counted |
| `:2027-2028` (in `run_sieve_pass`) | Replaces Stage 2's `validate_pwc_window_offset(offset)`: `validate_pwc_window_anchor(offset, ssh_route=self._tcp_transport is None)` |
| `:2159` (TCP) and `:2175-2176` (local Zeus) | The job is sent as `pwc_new_key_job(job, _window_anchor)`. The SSH branch (`_dispatch_to_worker`) still gets the old job with offset 0 |
| `:1858-1861` (`_dispatch_to_tcp`) and `:1929-1933` (`_dispatch_local_sieve`) | Echo check. A missing or different echo returns `status: error`, and Stage 1's completion accounting marks the chunk failed. The message contains neither "timeout" nor "pipe", so it is not retried as transient |
| `:2629-2632` (in `run_trial_persistent`) | Replaces Stage 2's trial-level refusal: `validate_pwc_window_anchor(config.offset, ssh_route=(pwc_transport != "tcp"))`, still before the fleet is built |

What the miner does and PWC now does: the coordinator sends both keys and no `offset`
(`range_miner_coordinator.py:2731-2732`). The anchor is checked against the post-session-filter domain and never clamped
(`range_miner_worker.py:913-925`; PWC: `sieve_filter.py` `PWC_WINDOW_ANCHOR_OUT_OF_DOMAIN`). The phase is pinned to 0.

### (b) The forward hybrids get the miner's argument counts (PWC new-key jobs only)

`sieve_filter.py`, `GPUSieve.run_hybrid_sieve`:
- **`:363`:** a new keyword argument, `miner_abi: bool = False`. Every non-PWC caller gets `False` and an unchanged launch.
- **`:376-383`:** for `miner_abi=True`, a forward hybrid whose kernel has no phase argument (`java_lcg`, `minstd`,
  `xorshift32`, `xorshift128`; miner `:160`) does not get the trailing `int32`. A non-zero phase for one of those is
  refused (`PWC_GENERATOR_PHASE_REFUSED`), as the miner refuses it (`:570-600`).
- **`:480`:** the trailing argument is `*([] if _drop_phase_arg else [cp.int32(offset)])`.
- **`execute_sieve_job` `:679`, `:723`:** passes `miner_abi=_pwc_new_schema`. A PWC job is always a new-key job now.

The resulting argument counts were measured on the real `run_hybrid_sieve` with CuPy replaced (G-HYB-1/2):

| Variant | PWC new-key | miner `EXPECTED_KERNEL_ARITY` | kernel's declared parameters | before (and legacy/ZMQ now) |
|---|---|---|---|---|
| java_lcg_hybrid | 15 | 15 | 15 | 16 |
| minstd_hybrid | 15 | 15 | 15 | 16 |
| xorshift32_hybrid | 16 | 16 | 16 | 17 |
| xorshift128_hybrid | 16 | 16 | 16 | 17 |
| lcg32_hybrid | 17 | 17 | 17 | 17 |
| pcg32_hybrid | 15 | 15 | 15 | 15 |
| all six `_hybrid_reverse` | 14 | 14 | 14 | 14 |

### (c) Old-style `offset` jobs refused at PWC's entry points; `load_draws_from_daily3` untouched

- **`persistent/pwc_worker_service.py` `_execute_job` (`:563-588`).** This is the TCP worker, PWC's entry point. The
  worker:
  - refuses a job carrying `offset` or `chunk_offset` (`PWC_LEGACY_OFFSET_REFUSED`);
  - refuses a job without both new keys (`PWC_WINDOW_ANCHOR_SCHEMA_REFUSED`);
  - never fills in a default `offset`; the defaulted `"offset"` line of the `sieve_job` dict is removed.

  Both refusals come back as `success: False` before any sieve work, as the miner's worker does (`:988`, `:1266`).
- **PWC's other entry point** is the local Zeus subprocess (`sieve_filter.py --job-file`). It only ever gets
  coordinator-built new-key jobs.
- **`sieve_filter.execute_sieve_job` still accepts old-style jobs.** Owner answer (c): the legacy `coordinator.py`
  (`:830`, run_trial's default path `window_optimizer_integration_final.py:2117`) and ZMQ (`zmq_sqlite_worker.py:111`)
  send them. G-OLD-1a–d shows those calls are unchanged.
- **`load_draws_from_daily3` is not edited.**

### (d) The stale checks

Already done by Stage 2. `tests/test_pwc_backup_step1.py` and `tests/test_pwc_stage1.py` pin `PRE_STAGE1_BASE =
"27eb991"`, the recorded pre-Stage-1 base, not HEAD. No further change was needed for (d).

One Stage-1 gate needed an update for (a), described under Tests.

---

## Step 3: hard limits, and the proofs

- **Untouched:** none of the ten files in `scripts/gate12_parity_gate.py:174-185`, nothing under `miner/`, no
  `window_optimizer*.py`. `git diff --stat 0caeaf9` lists only the files in "Files" below.
- **`sieve_filter.py`:** only `execute_sieve_job` and `GPUSieve.run_hybrid_sieve` changed, and RANGE-MINER reaches
  neither (above).

`prove_unchanged_outside_demo.txt`:

| File | Allowed to change | Result |
|---|---|---|
| `sieve_filter.py` | `execute_sieve_job`, `GPUSieve.run_hybrid_sieve` | exit 0: 31/31 top-level statements and the other 8 definitions identical |
| same, control | `execute_sieve_job` only | exit 1, names `GPUSieve.run_hybrid_sieve` |
| `persistent/pwc_worker_service.py` | `PWCWorkerService._execute_job` | exit 0, 20 other definitions identical |
| `persistent_worker_coordinator.py` | the 4 changed functions, plus 4 names declared new (`--allow-added`) | exit 0: 78 base statements identical and in order; the 4 additions are definitions and one constant |

**Import proof** (`scripts/pwc_sieve_filter_import_proof.py`, now for both modules RANGE-MINER's launch imports,
`window_optimizer_integration_final.py:52-58`). It runs a clean `python3 -I` subprocess with CuPy recorded, under the
real hostname and under `rig-6600`. Base and new are identical in:
- environment changes;
- every CuPy call, including `set_limit(268435456)` for `sieve_filter`;
- the set of modules loaded (161 for `sieve_filter`, 173 for the coordinator).

Files: `import_proof_sieve_filter.txt` (also `proof_output.json`) and `import_proof_persistent_worker_coordinator.txt`.
A deliberate import-time change in the coordinator (`os.environ.setdefault(...)`) fails both the AST check and the
import check (`logs/proof_pwc_mutant.json`).

The proof tool gained `--allow-added` (new top-level names must be absent from the base and present once in the new
file). Its tests are now 29/29, including a side effect placed next to an allowed addition. The import proof's AST
part now uses the tool, so it handles qualified names such as `GPUSieve.run_hybrid_sieve`.

---

## Step 4: tests

### Focused

| Suite | Result |
|---|---|
| `tests/test_pwc_stage2b_anchor_phase.py` (updated) | 35/35. New: G-OLD-1d, G-OLD-2 (worker refuses all 22 old-style jobs), G-WRK-1b/1c (refused by the worker), G-NEW-2d, G-HYB-1–5 (the real `run_hybrid_sieve` argument lists) |
| `tests/test_pwc_backup_stage2.py` (updated) | 26/26 checks, 33/33 mutants. See below |
| `tests/test_pwc_backup_step1.py` (updated) | 22/22 checks, 16/16 mutants, after the fix below (`focused_test_pwc_backup_step1_after_fix.txt`) |
| `tests/test_prove_unchanged_outside.py` | 29/29 |
| Mutants of the new `sieve_filter`/worker code | 5/5 caught (`mutants_sieve_filter_worker.txt`): single-phase hybrid not told `miner_abi`; trailing argument dropped for every caller; no-phase phase refusal removed; worker accepts `offset`; worker missing-key check removed |

Test edits needed by the change (both are PWC test files):
- **`tests/test_pwc_backup_stage2.py`:**
  - the fake worker results echo `window_anchor`/`generator_phase` when the job carries them, as `sieve_filter`'s
    result now does;
  - the real-`sieve_filter` echo gate gets a real dataset file (the anchor domain check reads it) and a
    `miner_abi`-aware stand-in;
  - `G-OFFSET` is replaced by `G-ANCHOR`:
    - TCP at anchor 25 runs, the jobs carry the split, and the pass is complete;
    - no echo, a wrong anchor or a wrong phase means no chunk counts;
    - `-1`, `"25"`, `2.0` and `True` are refused before dispatch;
    - SSH: 25 is refused before the fleet is built, a pass at 3 is refused, and a pass at 0 is not.
  - Mutants M-OF-1/2 are re-aimed at the new lines. M-AN-1–4 are new: TCP keeps `offset`; echo not checked; phase
    follows the anchor; negative anchor accepted.
- **`tests/test_pwc_backup_step1.py`:** `G-S2-ZERO-BYTE-EQUAL` compares the job dicts with the pinned `27eb991`. It
  already excluded the keys Stage 2 changed. A new-key job is now mapped back to `offset = window_anchor`, and only
  when `generator_phase` is 0. The rest of the dict must still be byte-identical.

### Full before / after

How the passes were run:
- **Runner:** `logs/pwc_stage2b_v1/run_suites.sh`, Stage 2's runner pointed at this worktree.
- **Suites:** Stage 2's 73 (no `tests/test_s172_staging_partb.py`), run one at a time, `timeout 1200` each.
- **Environment:** `~/venvs/torch`, CPU only, ssh/scp/rsync replaced by a refusing stub.
- **"Before":** clean `0caeaf9`, in this worktree. All Stage 2b files were parked in `logs/stage2b_parked/` (tar with
  sha256 manifest, checked before removal) and restored afterwards; status and `git diff` were byte-identical to the
  parked state.
- **"After":** this package, plus the 2 new suites. Run twice:
  - the **first after-pass** ran before the Step 1 test fix;
  - the **final after-pass** (`after_final`) ran on the final code, after the Beta relay, with no code change during or
    since it.

**Final result (`after_final`, the code as it lands): 71 of 73 existing suites are identical before and after (same
exit code and the same set of failing checks). 2 differ. Both new suites pass.**

| Suite | before | after_final | Why |
|---|---|---|---|
| `tests/test_s172_phase4_coordinator.py` | rc 0 | rc 1: Gate 22 | Git-status tripwire: it names exactly this package's uncommitted `.py` files. Transitional on an uncommitted tree (`docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`); clears on commit |
| `tests/test_s172_attempt6_remediation.py` | rc 1, 76/78 | rc 1, 75/78 (+`FAIR-1`) | See below |

**`FAIR-1`** (control-plane fairness, "count pressure") is a wall-clock bound:
- it measured T_cp = 0.878 s against a bound of 0.873 s in the first after-pass, and 0.929 s against 0.869 s in the
  final one;
- it **failed in 2 of 2 full after-passes**, and **passed in 1 of 1 before-pass and in 2 of 2 runs of the suite alone**
  on the changed tree (`logs/stage2b_focus/attempt6_rerun_{1,2}.log`, 76/78, the same failures as before);
- Stage 2's four passes had no FAIR-1 failure.

The suite references none of the changed files (0 hits for `sieve_filter`, `persistent_worker`, `pwc_`,
`window_optimizer`). Importing its only project module, `miner.range_miner_coordinator`, loads none of the four
changed modules. So no code path from this package into the check was found.

That it failed only in the full after-passes is a measured fact, not explained. Two full-pass samples are too few to
settle it. The measurement that would settle it: the same full pass repeated on clean `0caeaf9` and on this tree,
counting FAIR-1 results, or FAIR-1 run under a fixed CPU load on both trees.

Also:
- **`test_pwc_backup_step1.py`:** identical to "before" in the final pass (the Step 1 fix). In the first after-pass it
  was the third differing suite.
- **Phase-5 suites:** `test_s172_phase5_d1_engine`, `d1_workflow`, `d2_directional_uniqueness` and `d5_process_sharded`
  run the phase-4 suite inside themselves. So Gate 22 appears in their output too. Their exit codes (1 before, 1 after)
  and their own failing-check sets are unchanged.
- **`tests/test_s172_admission_stall.py`:** timed out at 1200 s in every pass, as it did in Stage 2's passes.
- **Live-state checks, all passes:**
  - `~/miner_staging`'s mtime was unchanged after every suite;
  - the main tree and the optuna listing were unchanged;
  - the ssh stub refused 25 calls in each pass, the same programs.
- **Code identity:** `after_final/code_sha256.txt` differs from `before/` only in this package's 5 tracked files and
  4 new files.

Files:
- `before_after_tally.md` and `before_after_failing_checks.txt`: the final pass;
- `before_after_tally_first_after.md` and `before_after_failing_checks_first_after.txt`: the first after-pass.

Logs: `logs/pwc_stage2b_v1/{before,after,after_final}/`.

---

## Beta round 1 (X-PWC-STAGE2B R1), relayed 2026-11-20

- **Request:** `docs/exchange/to_beta/X-PWC-STAGE2B_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`.
- **Review:** `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md`. Ledger row: `docs/exchange/LEDGER.md`,
  X-PWC-STAGE2B R1, 17:49 PST.
- **Relay verdict: BOUNDARY VIOLATION (exit 3).** The relay recorded 35 git-ignored files as newly created during the
  run:
  - `draws.json` and `replace.json`;
  - 12 files `results/window_opt_*_8_{0,1,3}_t0.json`;
  - 21 files under `stage2/`.

  Beta's review says it wrote them itself (`X-PWC-STAGE2B_R1_BETA_REVIEW.md:76`): its first fixture run imported
  Stage 2's harness, which changes the working directory to the repository.
- **Owner decision, 2026-11-20:** the verdict is **accepted as Beta's own fixture output**. The basis: after the run,
  the code under review (`git diff 0caeaf9` of the five tracked files, plus the four new files) was byte-identical to
  `pwc_backup_stage2b.diff` as written before the relay. The only other tracked changes were the relay's own rows in
  `docs/exchange/LEDGER.md` and `docs/exchange/DEFERRED.md`.
- **The 35 files** were moved, not deleted, on the owner's order, to `/tmp/beta_fixture_stage2b_20261120/`, keeping
  their paths.
  - Only paths the relay recorded as "ignored file created" were moved (35 of 35 recorded lines;
    `logs/relay_stage2b_created_paths.txt`).
  - No listed path existed before the relay. Every entry in `results/` and `stage2/` was on the list.
  - Hashes were checked after the move (`/tmp/beta_fixture_stage2b_20261120/MANIFEST.sha256`).
  - The emptied `results/` and `stage2/` directories were left in place.
- **Positions:** Q1, Q2, Q4, Q6, Q7 and Q9 NO OBJECTION; Q3 OBJECTION (THEORETICAL); Q5 and Q8 SUGGESTION; two
  findings. The review file is the record.
- **Outcome (lead, 2026-11-20):** **no `LIVE` objection.** No second Beta round.
  - **Q3:** THEORETICAL, deferred (`docs/exchange/DEFERRED.md`). It **matches RANGE-MINER's behaviour**: the miner
    worker (`miner/range_miner_worker.py:849`) and the miner coordinator (`miner/range_miner_coordinator.py:9322`,
    `:9337`) truncate with `int()` in the same way. A stricter PWC-only check is **not applied**, because the engines
    would differ. `Q3_PROPOSAL_NOT_APPLIED.md` keeps the design; any change must be made in both engines together.
  - **Carried to Stage 3:**
    - Q5: check trial completion before candidates are appended or the trial is scored (review `:59-61`);
    - Q8: limits of the proof tool, namely added names with import-time right-hand sides, rebinding an imported name,
      and allowed changes reaching unchanged callers (`:67`);
    - Finding 1: record EXEC-PIN-1 and the pinned-revision comparisons with the test edit (`:46`);
    - Finding 2: the two dataset reads in the new-key path (`:47`).
  - **Belongs to Stage 4:** Q4's GPU launch check of the new argument counts, and the hit-rate comparison on unseen
    draws (`:57`).
  - Nothing Beta raised was changed in this package.

## Files to land (exact list, 29 paths; from `git status --porcelain --untracked-files=all` at `0caeaf9`)

**Code, tracked, modified (3):**
- `sieve_filter.py`: `execute_sieve_job`, and `GPUSieve.run_hybrid_sieve` (`:363-383`, `:480`; `miner_abi` at `:679`, `:723`)
- `persistent/pwc_worker_service.py`: `_execute_job` only (`:563-588`); Stage 2's exit-43 change kept
- `persistent_worker_coordinator.py`: `:356-417` (new); `_dispatch_to_tcp` `:1858-1861`; `_dispatch_local_sieve`
  `:1929-1933`; `run_sieve_pass` `:2027-2028`, `:2159`, `:2175-2176`; `run_trial_persistent` `:2629-2632`

**Tests, tracked, modified (2):**
- `tests/test_pwc_backup_stage2.py`
- `tests/test_pwc_backup_step1.py`

**New code (4):**
- `scripts/prove_unchanged_outside.py`
- `scripts/pwc_sieve_filter_import_proof.py`
- `tests/test_prove_unchanged_outside.py`
- `tests/test_pwc_stage2b_anchor_phase.py`

**Exchange (4).** The ledger rows were written by the relay; the hashes of the two files match the ledger row:
- `docs/exchange/LEDGER.md` (+1 row, X-PWC-STAGE2B R1)
- `docs/exchange/DEFERRED.md` (+1 row, Q3)
- `docs/exchange/to_beta/X-PWC-STAGE2B_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md` (sha256 `9cd6115b…`)
- `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md` (sha256 `d54b142a…`; Beta's file, not edited)

**Package documents (16), all in `docs/drafts/pwc_backup_stage2b_v1/`:**
- `README.md`
- `Q3_PROPOSAL_NOT_APPLIED.md`
- `pwc_backup_stage2b.diff`, `joined_pwc_worker_service_vs_0caeaf9.diff`
- `before_after_tally.md`, `before_after_failing_checks.txt` (final pass)
- `before_after_tally_first_after.md`, `before_after_failing_checks_first_after.txt` (first after-pass)
- `prove_unchanged_outside_demo.txt`, `import_proof_sieve_filter.txt`, `import_proof_persistent_worker_coordinator.txt`
- `test_output.txt`, `test_prove_unchanged_outside_output.txt`, `focused_test_pwc_backup_stage2.txt`,
  `focused_test_pwc_backup_step1_after_fix.txt`, `mutants_sieve_filter_worker.txt`

**Not to land:**
- `docs/drafts/pwc_backup_stage2b_v1/proof_output.json`: git-ignored by `*.json` (`.gitignore:41`), and a copy of
  `import_proof_sieve_filter.txt`;
- `logs/`: ignored; runners, pass logs, backups;
- `data/`: untracked suite output;
- the empty `results/` and `stage2/` directories;
- `/tmp/beta_fixture_stage2b_20261120/`: Beta's fixture output, moved out of the tree.

**Untouched (verified):** none of the ten files in `scripts/gate12_parity_gate.py:174-185`, nothing under `miner/`, no
`window_optimizer*.py`, no kernels, no launcher.

---

## Deploy: the rigs and VM101 must change together

- **Rigs:** `sieve_filter.py` and `persistent/pwc_worker_service.py` go to each CT100 (`mkdir -p` first, absolute
  paths). Then `scripts/pwc_parity_gate.py --transport tcp` and `scripts/gate12_parity_gate.py`.
- **VM101:** the coordinator runs from the VM101 tree.
- **If only one side is updated, every PWC TCP pass fails, loudly:**
  - **new coordinator, old rig worker:** no echo, so every chunk fails (`pwc_anchor_echo_error`) and the trial is
    marked incomplete;
  - **old coordinator, new rig worker:** every job carries `offset` and is refused (`PWC_LEGACY_OFFSET_REFUSED`).

  Neither combination produces a silent wrong window.
- **Kernels and miner:** no kernel and no miner file changes.

## Not verified

- **Hardware:** any GPU run, and real CuPy/HIP with 15 instead of 16 arguments. The counts match each kernel's declared
  parameters, which is what the miner already launches.
- **Rigs:** real rigs and deployed bytes.
- **Completion handling:** that the caller of `run_trial_persistent` treats an incomplete trial as not counted. Stage 1
  marks it; who reads it is Stage 3.
- **Out-of-domain anchors in a real run:** what an Optuna-sampled anchor outside the domain does. Every chunk is
  refused and the trial is incomplete, by reading; not run end to end.
- **Owner decisions:** the 08:30 and 17:38 rulings and this session's three answers exist only as relayed text and this README.
- **FAIR-1:** why it failed in both full after-passes (see "Full before / after").

## Open questions (owner; not put to Beta as rulings)

Answered 2026-11-20 17:38 PST: the 08:30 ruling lifts the recorded Brief-II wait for PWC only.

1. **Owner/Stage 3:** the SSH route still cannot carry an anchor ≠ 0. Remove it (brief F1), or leave it at offset 0
   only?
2. **Owner/Stage 3:** should an out-of-domain anchor be refused on VM101 before dispatch, instead of on every rig
   chunk? That needs the dataset length on the coordinator side.
