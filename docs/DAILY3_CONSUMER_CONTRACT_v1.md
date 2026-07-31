# TEAM ALPHA → TEAM BETA — `daily3.json` consumer contract (v1)

**Purpose.** Establish, by tracing live code rather than by assumption, exactly what the
pipeline requires of the draw dataset, so a rewritten producer cannot silently break or
alter downstream steps before Beta freezes the publication schema.

**Nothing was changed.** Read-only investigation. No code, config, or data file was
modified. This document is the only artifact produced.

---

## 0. Scope declaration (VIR-6)

### Surfaces SEARCHED

| Surface | Method |
|---|---|
| Repo tree `/home/michael/distributed_prng_analysis` | `/bin/grep -rn` over all `.py`/`.sh`/`.md`/`.json` for `daily3(_midday\|_evening)?\.json`, `lottery_file`, `full_state`, `['draw']`, `get('session')`, `reversed(`, `[::-1]`, `[-N:]`. 741 `.py` files in scope (excluding `logs/`, `.git/`). |
| The three data files themselves | Full parse of all 18,068 + 8,515 + 9,553 records: key-set census, per-field type census, ordering, uniqueness, gap analysis, value range. |
| Git | `git ls-files`, `git log --all -S"preprocess_daily3"`, `git log --all -- '*preprocess_daily3*'`, `.gitignore`. |
| Host filesystem outside the repo | `find /` for `preprocess_daily3*`; `/home/michael/*.py`; `/home/michael/cluster_controller/`. |
| Host scheduling (VM 101 only) | `systemctl list-units --all \| grep -i daily3\|scraper`; `crontab -l` (michael); `sudo -n crontab -l` (root); `/etc/cron.d`; `systemctl list-timers --all`. |
| Live pipeline config | `optimal_window_config.json`, `config_manifests/parameter_registry.json`, `agents/watcher_agent.py` `FILE_VALIDATION_CONFIG`. |

### Surfaces UNAVAILABLE — unsearched, not clean

- **The three rig CT100s** (`.122`/`.156`/`.164`) and **bare-metal `.127`** — not swept for
  their own copies of the dataset, their own cron/timer wiring, or deployed-but-uncommitted
  consumers. `daily3.json` is known to exist on `.122` (hand-copied during Phase 6.0).
- **Windows VM 100** — RDP only.
- **Other git branches / deleted-file history** for consumers that no longer exist on `main`.
- **A whole-file semantic read of all 741 `.py` files.** Method was keyword + field-access
  pattern enumeration. A consumer that reaches a draw field through a fully computed key
  string with no literal on the line would evade it.
- **`/home/michael/Downloads`, `/home/michael/cluster_controller/.stversions`** — scanned
  only for the `preprocess_daily3` / `daily3_filtered` names, not enumerated as consumers.

---

## 1. The artifacts

| File | Location | Tracked? | Records | Span | mtime |
|---|---|---|---|---|---|
| `daily3.json` | repo root | **no** (`.gitignore:41` `*.json`) | 18,068 | 2000-01-01 evening → 2026-02-26 midday | Mar 4 16:58 |
| `daily3_midday.json` | repo root | no | 8,515 | 2002-11-04 → 2026-02-26 | Mar 6 17:54 |
| `daily3_evening.json` | repo root | no | 9,553 | 2000-01-01 → 2026-02-25 | Mar 6 17:54 |

**Derivatives produced by the pipeline itself** (not by the scraper):

- `train_history.json` / `holdout_history.json` — `window_optimizer.py:801-818` and
  `:991-1004`, an 80/20 **positional** split of `[d["draw"] for d in lottery_data]`.
- `daily3_midday.json` / `daily3_evening.json` — `dataset_split.py`.
- The Step-1 NPZ carries `window_size` and `offset` but **no dataset identity**
  (`tests/smoke_s172_phase5_d6_zeus_single_gpu.py:98-106`, the frozen 22-array list).

**NOT a derivative — do not conflate.** `lottery_history.json` is a **separate synthetic
lineage** with an incompatible schema: a `{"draws": [...], "metadata": {...}}` object whose
records carry `draw` as a **list of digits** plus `value`/`raw_value`, `draw_id`,
`timestamp`, `position`, `true_seed`, `draw_source`. It is written by
`draw_ingestion_daemon.py` and is not produced from `daily3.json`.

**`daily3_filtered.json`** — emitted by the legacy `preprocess_daily3.py` (§6). **Zero
consumers inside the repo**; the file does not exist anywhere on this host.

---

## 2. The record schema as actually consumed

Every one of the 18,068 combined records has **exactly** the key set
`{date, draw, session}` — one key-set, no variants, no nulls.

| Field | Type observed | Required in practice | Consumed by |
|---|---|---|---|
| `date` | `str`, `"YYYY-MM-DD"`, 100% well-formed | **Required** — `KeyError` in `per_draw_timestamp_sieve.py:106`; silent misbehaviour elsewhere | keyed lookup, filtering, reporting |
| `session` | `str`, lowercase, ∈ `{"midday","evening"}` | **Required** — a record whose `session` is absent/other is **dropped** by every session filter and by `dataset_split.py` | session filtering, split derivation |
| `draw` | `int`, range **0 … 999** (22 records are exactly `0`) | **Required** — `KeyError` in the sieve loaders | the residue value |
| `full_state` | `int` — **absent from all three production files** | **Optional, but load-bearing** | **Overrides `draw`** in every sieve loader |

### `full_state` — the field a new producer must not accidentally emit

```python
# miner/range_miner_worker.py:575  (and sieve_gpu_worker.py:115, sieve_filter.py:187,
# reverse_sieve_filter.py:117, sieve_filter_INTEGRATED.py:83,
# window_optimizer_integration_final.py:198)
return [int(entry.get("full_state", entry["draw"])) for entry in window]
```

If a record carries `full_state`, **the sieve silently uses it instead of `draw`**. Only
the synthetic-dataset generators (`create_synthetic_full_state.py`,
`variable_skip_dataset.py`, `create_fake_mt_dataset.py`) emit it. A new producer that
adds a field with this name — for any reason — silently replaces the residue stream.

### Unique identity of a draw

**`(date, session)`.** Verified unique across all three files (0 duplicates).
`evaluate_pools.py:23` and `backtest_pools.py:47` are the only consumers that look a draw
up by identity; both do a **linear scan returning the first match**, so a duplicated
`(date, session)` would be resolved silently to whichever copy is earlier in the array.

### Date/time format

`"%Y-%m-%d"`, zero-padded, no time component, no timezone. Two consumers parse it:
`preprocess_daily3.py:18` (`datetime.strptime(..., "%Y-%m-%d")`, hard failure on drift) and
`per_draw_timestamp_sieve.py:106` → `estimate_draw_time(date, session)`. Everything else
compares dates as **raw strings** (`backtest_pools.py:42`: `entry.get('date','') < cutoff`),
which is only correct because the format is lexicographically sortable.

---

## 3. Consumer inventory

### 3.A — Live pipeline (a change here changes results)

| # | Consumer | Fields accessed | Behaviour on missing/malformed |
|---|---|---|---|
| 1 | **`miner/range_miner_worker.load_residue_window`** `:538-575` — the D6 shared authority (§7) | `session` (filter), `full_state`→`draw` | `ResidueResolutionError` if `n < window_size`; **`KeyError` on a record with no `draw`** (uncaught → not the clean `ResidueError` the protocol expects); records with an unmatched `session` are **silently dropped** |
| 2 | `sieve_gpu_worker.load_draws_cached` `:102-118` | same | `ValueError` on short dataset; `KeyError` on missing `draw`. **Caches on `(path, window_size, sessions, offset)`** — a rewritten file at the same path is served stale within a process |
| 3 | `sieve_filter.load_draws_from_daily3` `:174-188` (+ `reverse_sieve_filter.py:106`, `sieve_filter_INTEGRATED.py:66`, `reverse_sieve_filter_INTEGRATED.py:49`) | same | `ValueError` on short dataset; `KeyError` on missing `draw` |
| 4 | `window_optimizer_integration_final._miner_residues_for_config` `:202-225` (RANGE-MINER parent) and `_get_residues_for_config` `:178-199` (PWC/ZMQ parent) | via 1 and 3 | inherits |
| 5 | **`window_optimizer.py` (Step 1)** `:801-818`, `:991-1004` | `draw` only, via `if isinstance(lottery_data[0], dict) and "draw" in lottery_data[0]` | If the **first record** lacks `draw`, the whole array is passed through as-is and downstream integer maths fails far from the cause |
| 6 | **`full_scoring_worker.load_lottery_history`** `:201-224` (Step 3) | tries `draw`, then `number`, `value`, `result` — **keyed on the first record only** | `ValueError` if none present; a heterogeneous array is read using the first record's key for every record → `KeyError` |
| 7 | `prediction_generator.py:1052-1059` (Step 6) | `d['draw']` | `KeyError` |
| 8 | `reinforcement_engine.py:345-411` | receives `List[int]` from 6/7 | n/a |
| 9 | `trse_step0.load_draws` `:769-782` (Step 0) | `d["draw"]` | `FileNotFoundError`; `ValueError` on non-list/empty; `KeyError` on missing `draw` |
| 10 | `agents/watcher_agent.py` `_validate_json_structure:203-238` / `evaluate_file_exists:283-306` | **none** | The **only** gate the dataset passes through. Checks: parses as JSON, not `null`, array non-empty, file ≥ 50 bytes. **No `daily3*` entry exists in `json_array_minimums` (`:168-175`)** — a 1-record `daily3.json` passes preflight |
| 11 | `dataset_split.py` (manual, §5) | `session`, `draw`, `date` | Unknown `session` → warn + **exclude from both outputs**; non-dict first record → positional fallback |
| 12 | `backtest_pools.py:27-49` → `watcher_kpi_baseline.py` | `date`, `session`, `draw` | `get(...)` with defaults; a missing `draw` → `int(None)` `TypeError` |
| 13 | `evaluate_pools.py:17-26` | `date`, `session`, `draw` | Returns `None` if `(date, session)` not found — **the caller cannot distinguish "no draw yet" from "dataset broken"** |
| 14 | `coordinator.py:1876-1885` | `number` → `draw` → `value` | **Falsy-zero defect** (§8.1) |
| 15 | `modules/window_optimizer.py:54,110,305` | passes `'daily3.json'` as `current_target_file` | n/a |
| 16 | `agents/pipeline/pipeline_step_context.py:69` | declares `required_inputs=["daily3.json"]` for Step 0 | declaration only, not enforced |

### 3.B — Analysis / validation tools (reachable, not on the Step 0-6 path)

`validate_survivors.py:155-174` (`draw`, `session`, **`reversed(data)`** — §8.2) ·
`prng_classifier.py:659-675` (`draw`, else `number`) ·
`digit_sequential_sieve.py:143-180` (`session`, then `draw`→`value`→`result`→`number`;
**falsy-zero defect**, §8.1) · `per_draw_timestamp_sieve.py:87-108` (`date`, `session`,
`draw`, all required; **`filtered[-30:]`**) · `model_health_check.py:386` ·
`historical_analysis_real.py:95,170,223-224` (**`lottery_data[0]` = earliest,
`[-1]` = latest**) · `trse_entropy_probe.py:63` · `trse_calibration_probe.py:475` ·
`w8_correlation_test.py:75` · `autocorrelation_probe.py` · `ds_cross_compare.py:37` ·
`unified_system_working.py:539-544,666,697` (`get('draw') is not None` — the only
zero-safe extractor found) · `modules/direct_analysis.py:425-433` (**`data[-count:]`**) ·
`modules/advanced_research.py:661-662` (**`lottery_data[-20:]`**) ·
`investigate_prng_bias.py` · `machine_fingerprint_probe.py` · `analyze_my_lottery_data.py` ·
`final_cracker.py` · `discover_custom_prng.py` · `explore_shift17_clue.py` ·
`test_reinforcement_with_real_survivors.py`.

### 3.C — Harnesses that pin the contract

`tests/smoke_s172_phase5_d6_zeus_single_gpu.py:107,536-537` (uses the **real**
`daily3.json`, `offset=0`, `sessions=["midday","evening"]`, and shells
`sha256sum daily3.json` at `:1008`) · `tests/test_s172_phase5_d6_threshold_path.py` (calls
`load_residue_window` directly at `:224`, `:413`) · `tests/test_s172_phase4_coordinator.py` ·
`tests/test_s172_phase1_scaffolding.py` · `test_persistent_worker_harness.py:729`
(synthesises `{"draw": i, "full_state": i*10}`).

---

## 4. Ordering and indexing — the highest-risk surface

**Array order is semantically load-bearing in four independent ways.** No consumer
validates it, and no consumer can detect a change in it.

### 4.1 What the file actually is

- **Sorted ascending by `(date, session)` as a raw tuple** — verified: `keys == sorted(keys)`.
- Because `"evening" < "midday"` lexicographically, **within a date the evening draw
  precedes the midday draw** — in all 8,514 dual-session dates. This is **chronologically
  inverted** and is an artifact of the sort key, not of the draw times.
- The known producer, `/home/michael/daily3_scraper.py:87-91`, emits
  `for year: for draw_type in ["midday","evening"]` — grouped by year then session. **It
  does not produce the order the file is in.** The current ordering was imposed by some
  step that is not in the repo and not in git history.

### 4.2 Index = position in the PRNG output stream

`full_scoring_worker.compute_holdout_hits_batch:284-306`:

```
CRITICAL: offset is DERIVED from train_history_len, not configurable.
holdout data is positions [train_history_len : train_history_len + holdout_len]
...
offset = train_history_len          # "OFFSET DERIVED - THIS IS THE LAW (per Team Beta)"
predictions = prng_func(seed, n_holdout, skip=offset)
```

The concatenation `train_history + holdout_history` — i.e. **the raw array order of
`daily3.json`** — is treated as the generator's output sequence, and the array index is
the advance count. Any insertion, deletion, dedup, backfill, or re-sort changes the
meaning of every index at or after the change point and silently invalidates
`holdout_hits`, which is the input to `holdout_quality`, the ML target.

`prediction_generator.py:839` (`next_idx = len(lottery_history)`) makes the same
assumption for the forward prediction position.

### 4.3 `offset` slices from index 0 — i.e. from the **oldest** end

Every loader does `start = max(0, min(int(offset), n - window_size))`, then
`data[start:start + window_size]`. With the live `optimal_window_config.json`
(`window_size: 21`, `offset: 66`, both sessions) against the current 18,068-record file,
the production residue window is:

```
data[66:87]  →  2000-03-07 evening (784)  …  2000-03-27 evening (849)
```

**The production sieve analyses draws from March 2000.** That is a direct consequence of
(a) the file being oldest-first and (b) `offset` being a head-relative slice. A producer
that emitted newest-first would leave every config file, every `residue_sha256`, and every
harness syntactically valid while silently switching the entire system onto recent data.

Note also `config_manifests/parameter_registry.json:38-43` describes `offset` as *"advance
seeds by offset*(skip+1) before testing"* — which is **not** what any loader does. The
loader is authoritative; the registry description is stale.

### 4.4 Reverse-sieve direction is derived from array order

`miner/range_miner_worker.py:813` — `residues[::-1] if reverse else residues`
(also `sieve_gpu_worker.py:189`, `sieve_filter.py:232,395`). The reverse pass is
literally the on-disk window reversed. Changing file order changes what "reverse" means.

### 4.5 Two contradictory intra-day orderings coexist

| Consumer | Intra-day order | Effect |
|---|---|---|
| On disk / all sieve loaders / Step 1 / Step 3 | **evening, then midday** | the order everything numeric is computed against |
| `backtest_pools.load_dataset:33` — `session_order = {"midday": 0, "evening": 1}` | **midday, then evening** | re-sorts to the chronologically correct order before backtesting |
| `validate_survivors.load_target_dataset:170` — `for entry in reversed(data)` | reverses the whole file | §8.2 |

`backtest_pools.py` is the **only** consumer that normalises. It therefore evaluates
against a different sequence than the pipeline it is measuring.

### 4.6 Positional-tail consumers ("the last N draws")

`per_draw_timestamp_sieve.py:98` (`filtered[-30:]`) ·
`modules/direct_analysis.py:429` (`data[-count:]`) ·
`modules/advanced_research.py:662` (`lottery_data[-20:]`) ·
`historical_analysis_real.py:223-224` (`[0]` = earliest, `[-1]` = latest).
All assume **tail = newest**. All break silently under a reversed or regrouped producer.

---

## 5. Session handling and the split files

### 5.1 How a session is distinguished

Solely by the literal lowercase string in `record["session"]`. There is no time field, no
draw-time, and no other discriminator anywhere in the record.

Every session filter in the system is the same one-liner —
`[e for e in data if e.get("session") in sessions]`
(`range_miner_worker.py:567`, `sieve_gpu_worker.py:109`, `sieve_filter.py:180`,
`per_draw_timestamp_sieve.py:95`, `backtest_pools.py:40`) — using `.get()`, so **a record
with a missing, capitalised, or renamed `session` is silently excluded** from every
single-session trial and from both split files, with no count reported and no error.

`sessions` is a first-class trial parameter: `window_optimizer.py:156-160` searches over
`[['midday','evening'], ['midday'], ['evening']]`, and it is persisted per trial in the
miner ledger (`range_miner_coordinator.py:526,1413`).

### 5.2 How `dataset_split.py` derives the split files

`dataset_split.py:36` hardcodes `SOURCE = Path("daily3.json")` and must be run from the
repo root. `:50-61` — if `raw[0]` is a dict containing `session`, partition on
`r.get('session') == 'midday'` / `== 'evening'`; anything else is warned about and
**excluded from both files**. `:78-79` writes both with `indent=2`. `daily3.json` is not
modified.

**Order is preserved** by the list comprehension, so the split files inherit the combined
file's ordering — and because each is single-session, the evening-before-midday anomaly
disappears and each split file is genuinely chronological.

`:63-72` — a **positional fallback** (`raw[0::2]` = midday, `raw[1::2]` = evening) if the
first record is not a dict with `session`. **This fallback would produce wrong data
against the current file even if `session` were dropped**, because 2000–2002 is
evening-only (2,939 evening draws before the first midday draw on 2002-11-04) and because
within a date evening precedes midday. It is unreachable today; it must not be relied on.

### 5.3 Who consumes which

- **Combined `daily3.json`** — the default everywhere: `optimal_window_config.json`
  currently has `sessions: ["midday","evening"]`; Step 0/1/3/6, all smoke tests, all
  probes, and every hardcoded default (`modules/window_optimizer.py:305`,
  `persistent_worker_coordinator.py:1850`, `integration/sieve_integration.py:442`,
  `system_core.py:43`) name `daily3.json`.
- **Split files** — consumed only when passed explicitly as `lottery_file`:
  `watcher_agent.py --run-pipeline --params '{"lottery_file": "daily3_midday.json"}'`
  (`dataset_split.py:21-28`). **No code path selects them automatically.** No file in the
  repo references `daily3_midday.json` or `daily3_evening.json` other than
  `dataset_split.py` and documentation.

### 5.4 The S119 claim — CONFIRMED, and unenforced

`SESSION_CHANGELOG_20260307_S119.md:132`:

> `dataset_split.py` committed to both repos — **run after each scraper refresh**

**Confirmed as a manual step**, and confirmed that *nothing automates or verifies it*:

- Repo-wide search for `dataset_split` outside itself returns only documentation and two
  comment lines in `pa_pick3_scraper.py:34-36`. There is no invoker in any `.py`, `.sh`,
  manifest, or `STEP_SCRIPTS` map.
- No crontab for `michael`; **no crontab for root**; `/etc/cron.d` stock only; 18 systemd
  timers, none project-related.
- The one project scraper unit, `daily3scraper.service`, is `loaded failed failed` — its
  target `run_daily3scraper.py` does not exist (Finding C, `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md:63-92`).
- Consistent with the current mtimes: `daily3.json` Mar 4 16:58, both splits Mar 6 17:54.

**Consequence for the new producer:** the split files are *not* regenerated by anything.
After a scrape they are stale until a human runs `dataset_split.py`, and **no consumer
detects the staleness** — there is no cross-file digest, no record-count check, no mtime
comparison anywhere in the pipeline.

Two further defects in the surrounding documentation:

- `pa_pick3_scraper.py:35` and `docs/SESSION_CHANGELOG_20260314_S143.md:240` instruct
  `python3 dataset_split.py --source pa_pick3.json`. **`dataset_split.py` has no argparse
  and no `--source`** — the invocation would fail, or (worse, since Python ignores unknown
  `sys.argv` here) it would silently re-split `daily3.json` and overwrite the CA splits.

---

## 6. `preprocess_daily3.py`

**It does not sit between the scraper and the pipeline. It is not in the pipeline at all.**

- **Location: `/home/michael/cluster_controller/preprocess_daily3.py`** — outside the repo,
  in a pre-TFM working directory dated Aug 2025. It is **not in `git ls-files`, not in
  `git log --all`**, and no copy exists in the TFM tree. It is the file referenced at
  `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md:140`.
- **What it transforms:** reads `./daily3.json`; for each record parses
  `datetime.strptime(record["date"], "%Y-%m-%d")` and lowercases `record["session"]`;
  drops any record whose session is not `midday`/`evening`; drops **midday draws before
  2008**; passes everything else through **unmodified**.
- **What it emits:** `daily3_filtered.json` — same record shape, `indent=2`.
- **Who consumes its output:** inside the TFM repo, **nobody**. `daily3_filtered.json` is
  referenced only by six other files in `/home/michael/cluster_controller/`
  (`ml_predict_delta.py`, `verify_seeds.py`, `analyze_verification.py`,
  `analyze_offsets.py`, `gpu_scanner_cupy.py`, `gpu_scanner_narrow.py`) and one copy in
  `/home/michael/Downloads/`. The file itself does not exist anywhere on this host.
- Its docstring at `:10` — *"Run daily3_scraper.py first"* — is the only cross-reference to
  `/home/michael/daily3_scraper.py`, which is likewise uninvoked.
- Its 2008 midday cutoff is **superseded by the data**: the actual first midday draw is
  **2002-11-04**, and the pipeline uses all 8,515 midday records.

**Recommendation:** treat `preprocess_daily3.py` as archaeology from the pre-TFM
`cluster_controller` prototype. It defines no requirement the new producer must satisfy,
and `daily3_filtered.json` should not be reintroduced.

---

## 7. `load_residue_window(path, window_size, sessions, offset)` — the correctness-critical consumer

`miner/range_miner_worker.py:538-575`. Per the D6 correction (Beta §4) this is **the**
session-aware residue derivation; both sides of every stripe assignment call it:

- the **worker**, via `ResidueResolver._loader` (`:606`), to rebuild the window it sieves;
- the **coordinator**, via `window_optimizer_integration_final._miner_residues_for_config`
  (`:223-225`), to build the residues whose `sha256` is stamped into the payload.

`_load_window_fresh` (`:580`) is an alias to the same object.

### What it requires of the dataset structure

1. **Top level is a JSON array.** `json.load` then `len(data)` and slicing. A `{"draws": …}`
   object raises `TypeError`/`KeyError`, not a `ResidueError`.
2. **Records are objects** — `e.get("session")`, `entry.get("full_state", entry["draw"])`.
   A bare-integer array (accepted by `_get_residues_for_config:198` and by
   `trse_step0.py:779`) raises `AttributeError` here.
3. **`session` matches exactly** — case-sensitive `in sessions` against the trial's
   `["midday"]` / `["evening"]` / `["midday","evening"]`.
4. **`draw` exists on every record in the selected window** — `entry["draw"]` is a plain
   subscript. A missing key raises `KeyError`, which is **not** a subclass of `ResidueError`,
   so it does not become `stripe_error(retryable=False)`; it escapes as an unclassified
   worker crash.
5. **`draw` (or `full_state`) is `int()`-able** and, for the java_lcg kernels, must fit the
   `cp.uint32` residue array (`:813`). Values 0…999 satisfy this; a negative or `null`
   value would not.
6. **`len(data)` after filtering ≥ `window_size`**, else `ResidueResolutionError`.
7. **Order is the file's order.** No sort, no reversal, no date awareness. `offset` is a
   head-relative index into the filtered array (§4.3).
8. **No caching by pathname** — deliberate (Beta clarification 1): the file is re-read on
   every call so a changed file is never served stale. `sieve_gpu_worker.load_draws_cached`
   does **not** share this property.

### The integrity chain built on top of it

`ResidueResolver.resolve` (`:610-669`) requires **mandatory `dataset_sha256`** on every
assignment (TB Blocker-6, Option C) and verifies it against the locally computed file
digest **before** any cache return, then verifies `sha256_residues(residues)` against the
payload's `residue_sha256`. `sha256_residues` (`:523-527`) hashes
`json.dumps([int(x) …], separators=(",",":"))` — i.e. **the ordered residue list**, so the
fingerprint is order-sensitive by construction.

**Direct consequence for a rewritten producer:** any byte-level change to the dataset —
including a whitespace/indent change, a key-order change, or an append — changes
`dataset_sha256`. Under the current code the digest is computed at
`range_miner_coordinator.py:3499` **per `serve_range` call**, not frozen per run. A scrape
landing mid-run puts nodes on different bytes; the freeze-at-run-start correction proposed
in `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md` §A is **not yet implemented** and remains
awaiting Beta ruling.

---

## 8. Implicit invariants nothing validates

The **only** validation the dataset ever passes is
`agents/watcher_agent.py:203-238` — parses as JSON, not `null`, array non-empty, ≥ 50
bytes. There is **no `daily3*` pattern in `json_array_minimums`**, so a truncated file
passes. Everything below is depended on and unchecked.

| # | Invariant | Who depends on it | What breaks, silently |
|---|---|---|---|
| 1 | **`draw` may be `0`** and must survive extraction | 22 real records | `digit_sequential_sieve.py:161-162` and `coordinator.py:1881` use `entry.get("draw") or entry.get(…)` — **`0` is falsy**, so those records fall through to `None` and are **dropped**. Off-by-22 residue streams |
| 2 | **File is oldest-first ascending** | §4.2, §4.3, §4.6 | `validate_survivors.py:170` does `for entry in reversed(data)` with the comment `# Oldest first` — it believes the file is **newest-first**. Its "chronological" sequence is reversed. `SESSION_CHANGELOG_20260228_S113.md:29` ("80/20 chronological split") confirms ascending is the intended convention, so this is the outlier |
| 3 | **Intra-day sort key is `(date, session)` raw** (evening first) | all residue derivation | Switching to true chronological (midday first) changes every combined-session residue window and every `residue_sha256`, with no error anywhere |
| 4 | **`(date, session)` is unique** | `evaluate_pools.py:23`, `backtest_pools.py:47` | A duplicate resolves to the first hit; the second is invisible. KPI hit/miss silently attributed to the wrong draw |
| 5 | **Prior content is stable across scrapes** (append-only in effect) | §4.2 index-as-position | A rewrite that corrects, reorders, or dedups a historical draw invalidates every previously computed `holdout_hits` with no signal |
| 6 | **`session` strings are exactly `"midday"`/`"evening"`, lowercase** | every filter, `dataset_split.py` | `"Midday"`, `"MID"`, `"eve"` → records vanish from single-session trials and from both split files, with only `dataset_split.py` printing a warning |
| 7 | **`date` is `YYYY-MM-DD`, zero-padded, lexicographically sortable** | `backtest_pools.py:42` string comparison | `"2026-2-5"` or `"02/05/2026"` sorts wrongly; train/test leakage in backtests |
| 8 | **`draw` is an `int`, not a string, not a digit list** | all extraction | `"090"` → `int()` works but `full_state` semantics and `% 1000` change; `[0,9,0]` (the `lottery_history.json` shape) → `TypeError` |
| 9 | **`draw` ∈ 0…999** (`mod = 1000` everywhere) | `validate_survivors.py:171` (`% 1000`), `full_scoring_worker.py:268,307` | A 4-digit value is silently truncated by `% 1000` |
| 10 | **No `full_state` key** | §2 | Adding it silently replaces the residue stream |
| 11 | **Record count grows monotonically at the tail** | Step-1 80/20 split; Step-3 holdout offset | Prepending or backfilling shifts the split boundary and every holdout position |
| 12 | **Date gaps are tolerated but positions are not** | everything positional | The dataset already has one gap (midday **2019-01-25** missing; evening has none). No consumer detects gaps — positions, not dates, are what count |
| 13 | **Split files are consistent with the combined file** | anyone passing `lottery_file` | Nothing compares them. §5.4 |
| 14 | **The file is at a path relative to the repo root** | `dataset_split.py:36`, `system_core.py:43`, `persistent_worker_coordinator.py:1850`, `digit_sequential_sieve.py:200,265` (which defaults to **`data/daily3.json`** — a directory that **does not exist**) | Path drift |

---

## 9. Specification for the new producer

### MUST — hard requirements

1. Emit a **JSON array at the top level**. Not an object, not NDJSON, not a `{"draws": …}`
   wrapper.
2. Every element is an **object** with **exactly** the keys `date`, `session`, `draw`.
3. `date` — string, **`YYYY-MM-DD`**, zero-padded, no time, no timezone.
4. `session` — string, **exactly `"midday"` or `"evening"`**, lowercase, no other value.
5. `draw` — **JSON integer**, `0 ≤ draw ≤ 999`. **`0` is valid and must be emitted as `0`**,
   not `"000"`, not omitted.
6. **`(date, session)` is unique across the array.** No duplicates, ever.
7. **Sort ascending by the raw tuple `(date, session)`** — which places **evening before
   midday within a date**. This is the order the entire numeric pipeline is computed
   against. Do **not** "fix" it to true chronological order without a governed migration
   (it changes every `residue_sha256`, the Step-1 split boundary, and every `holdout_hits`).
8. **Historical records are immutable.** A refresh appends at the tail; it does not
   reorder, renumber, dedup, or correct prior rows. If a correction is genuinely required,
   it is a dataset-version event, not a scrape.
9. **Write atomically** (temp file + `os.replace`). A partial write is a valid-JSON-shaped
   truncation that passes WATCHER's preflight (§8) and silently shortens every window.
10. **Never emit a key named `full_state`** (§2).

### MUST NOT

11. Do not emit `null` for any field.
12. Do not emit extra fields casually. Nothing rejects them, but each becomes a new
    implicit surface, and `full_state` in particular is actively load-bearing.
13. Do not change indentation/serialisation style casually — it changes `dataset_sha256`
    and invalidates every in-flight residue check (§7).
14. Do not write the split files. That is `dataset_split.py`'s job (§5).

### SHOULD — to close the gaps this trace exposed

15. Support an explicit **append** mode that reads the existing file, appends only
    genuinely new `(date, session)` pairs, and re-sorts by `(date, session)` — rather than
    the current scraper's full-rewrite (`daily3_scraper.py:99`), which is what makes
    invariant 8 unenforceable.
16. Emit a **summary to stdout**: records before/after, records appended, first/last
    `(date, session)`, and any `(date, session)` collision. Today a scrape reports nothing
    an operator could diff.
17. **Do not silently drop rows.** The current scraper's `continue` on a parse failure
    (`daily3_scraper.py:56,68,71`) produces a shorter file with no non-zero exit — which is
    indistinguishable downstream from a shorter history.
18. **Fail loudly and leave the old file in place** if the new content has *fewer* records
    than the existing file, unless explicitly forced.
19. Either **regenerate the split files** as part of the refresh, or have the refresh emit
    a marker the pipeline can check — because §5.4 establishes nothing else will.

### Decisions Beta must make (they are not derivable from the code)

20. **Ordering convention is currently an accident, not a decision.** No committed producer
    generates the order the file is in, and the code base contains three mutually
    inconsistent beliefs about it (§4.5, §8.2). The publication schema should state the
    convention normatively.
21. **`offset` is head-relative** (§4.3), so under the live config the sieve analyses
    March 2000. If the intent is recent draws, that is a *config* correction, not a
    producer change — but a producer that flips the ordering would appear to "fix" it while
    breaking everything else.
22. **Dataset versioning.** Findings A and B of `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md`
    (freeze-at-run-start; record the frozen digest in the generation sidecar) are still
    awaiting ruling and become load-bearing the moment scraping is automated.
23. **Cross-file consistency.** There is no mechanism binding `daily3_midday.json` /
    `daily3_evening.json` to the `daily3.json` they were split from.

---

## 10. Defects observed during this trace (reported, not fixed)

| # | Location | Defect |
|---|---|---|
| D1 | `digit_sequential_sieve.py:161-162`; `coordinator.py:1881` | `entry.get("draw") or …` drops the 22 records where `draw == 0` |
| D2 | `validate_survivors.py:170` | `reversed(data)` commented `# Oldest first` — the file is already oldest-first, so the sequence is reversed |
| D3 | `pa_pick3_scraper.py:35`; `docs/SESSION_CHANGELOG_20260314_S143.md:240` | Instruct `dataset_split.py --source …`; the script has no argparse and hardcodes `SOURCE = Path("daily3.json")` |
| D4 | `config_manifests/parameter_registry.json:38-43` | `offset` described as *"advance seeds by offset*(skip+1)"*; every loader treats it as a head-relative array slice index |
| D5 | `digit_sequential_sieve.py:200,265` | Defaults to `data/daily3.json`; no `data/` directory exists |
| D6 | `agents/watcher_agent.py:168-175` | `json_array_minimums` has no `daily3*` entry — a 1-record dataset passes preflight |
| D7 | `dataset_split.py:63-72` | Positional fallback (even=midday / odd=evening) would produce wrong data against this dataset's actual shape; unreachable today |
| D8 | `range_miner_worker.py:575` | `entry["draw"]` raises bare `KeyError`, not a `ResidueError`, so a malformed record escapes the `stripe_error(retryable=False)` classification the protocol expects |

**None of these were changed.** D1, D2, and D8 are the three that would corrupt results
rather than crash.
