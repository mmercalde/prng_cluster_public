# CLAUDE CODE REPORT — STEP 6 → CHAPTER 13 GRADING CONTRACT, STAGE A (READ-ONLY)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_STEP6_CH13_GRADING_CONTRACT_STAGE_A_20261113.md`
**Binding addendum applied:** `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` (§A, §B, §E)
**Host:** VM 101 (`zeus-ubuntu`), user `michael`. **Worktree:** `/home/michael/tfm_worktrees/wp2_stage_a`
(detached at `0066b6e`). **Date:** 2026-11-13. **Package:** WP-2. **Relay thread:** X-0003.
**Step numbering:** EXECUTABLE scheme (Step 6 = `prediction_generator.py`; Chapter 13 is not a WATCHER step).
**Yardstick:** this report asks what is written and what is read. It makes no claim about whether TFM predicts.

---

## 0. First-step output (verbatim, 2026-11-13T18:54:01-08:00)

```
--- pgrep -af "range_miner_coordinator|watcher_agent|window_optimizer"
11539 /bin/bash -c source /home/michael/.claude/shell-snapshots/... eval 'date -Is; ... pgrep -af "range_miner_coordinator|watcher_agent|window_optimizer"; ...'
rc=0                       <- the only match is the probing shell itself (pattern self-match, CLAUDE.md rule 8)
--- ss -ltn | grep 5700
rc=1                       <- nothing listening on 5700
--- git -C /home/michael/distributed_prng_analysis status --porcelain
(empty)
--- git worktree list
/home/michael/distributed_prng_analysis  0066b6e [main]
/home/michael/tfm_worktrees/rm_gates     fbd8264 (detached HEAD)
HEAD = 0066b6e9a2d701a41d26a22e7e338f9073104718
```

Self-match-free re-checks (`ps -eo pid,args | /bin/grep -E … | /bin/grep -v grep`) returned no process
immediately after worktree creation, again before the SQLite read (§Q6), and at 19:07:24 before writing this
report. No run was live at any point; nothing heavier than small CPU scripts was run.
After `git worktree add --detach /home/michael/tfm_worktrees/wp2_stage_a HEAD`, `git worktree list` also showed
`/home/michael/tfm_worktrees/wp1_stage_a 0066b6e (detached HEAD)`: another session's worktree, not created by me
and not touched. No file in play was outside the no-touch list (main tree clean).

**Base drift:** the brief's anchors were read at `26d3147`. `git diff --stat 26d3147 HEAD -- prediction_generator.py
chapter_13_diagnostics.py chapter_13_orchestrator.py agents/watcher_agent.py` is empty, so every brief anchor holds
at `0066b6e`. All anchors below were opened in this session at `0066b6e`.

---

## 1. Summary (ten lines)

1. **Null result: no Step 6 pool has ever been graded against a real draw on this host.** That covers 233/233
   archived Chapter 13 diagnostics (all synthetic draws, all `pool_size 0`), 0/56 Step 1 DB rows carrying a
   `downstream_score`, and 12/12 `backtest_temp` metrics files (real dates, but `survivor_count 0` and empty
   xorshift32 pools, Oct 2025).
2. Step 6 writes `predictions/next_draw_prediction.json` plus a timestamped archive copy. `predictions` is a list
   of at most `k` (20 under WATCHER) **integers**, with parallel score arrays. No 100- or 300-entry pool exists
   anywhere in Step 6.
3. Chapter 13 reads `prediction_pool.json` at the project root, which nothing writes. It expects a list of
   **objects**, and a missing file silently becomes an empty pool.
4. Q3 reproduced Beta's control. The object pool returns 1 hit. The integer pool, the real host artifact, and
   all 19/19 archived Step 6 outputs raise `AttributeError: 'int' object has no attribute 'get'`.
5. Step 6 records **no target-draw identity**: no date, session, draw id, history length or dataset hash. It
   collapses the combined-session dataset to bare integers and forecasts combined index `len(history)`.
6. Chapter 13 grades against `draws[-1]` of `lottery_history.json`, a separate, synthetic-only file. A draw
   appended after the pool was made is the one graded. The two cannot be matched today.
7. Missing pools, `PREDICTION_SKIPPED` abstentions and stale files are graded as **misses**. A missing
   `timestamp` is filled with `now()`. Malformed values raise exceptions.
8. Duplicates are real, not hypothetical: 4/19 archived Step 6 pools hold 20 entries but 19 distinct values. The
   grader counts a duplicated winner as `exact_hits=2`, and `best_rank` (which feeds hit@20/100/300) also counts
   ±5 near hits.
9. Downstream, the grade reaches the triggers, the LLM advisor and acceptance. Through the S140b write-back it
   also reaches `prng_analysis.db.downstream_score`, and that column ranks Step 1 warm-start parameters on the
   `n_parallel > 1` path.
10. Q8 sets out three landing options (A: Step 6 only; B: Step 6 + Chapter 13; C: Step 6 + a standalone frozen
    grader). All of them build on the already-approved S179 §2.5/§2.6/§6 ranked-unique artifact. None needs
    `agents/watcher_agent.py` or `agent_manifests/` for the minimum contract. Nothing is implemented.

---

## Q1 — Everything Step 6 writes

**Paths.** `_save_predictions` (`prediction_generator.py:896-952`) writes:

| path | role | write mode |
|---|---|---|
| `predictions/next_draw_prediction.json` | canonical ("WATCHER contract", `:902-907`) | plain `open(...,'w')` + `json.dump` (`:945-946`), **not atomic** |
| `predictions/history/predictions_YYYYMMDD_HHMMSS.json` | archive, "non-contractual" (`:909-913`, `:949-951`) | same content, plain write |

Both paths are **CWD-relative** (`Path("predictions")`, `:905`). `--output-dir` sets `config.predictions_dir`
(`:1022-1023`), which `_save_predictions` never reads, so the CLI output directory has no effect. Both files are
gitignored (`.gitignore:41 *.json`, verified with `git check-ignore -v`). Nothing else is written: the
`--test` and `--dry-run` paths write nothing (`:990-1037`).

**When a file is written** (three result shapes, all through `_save_predictions` when `save_predictions` is true):

| case | anchor | shape |
|---|---|---|
| success | `:709-738` | see schema below |
| signal gate blocked in `__init__` | `:333-341`, saved `:585-587` | `{status:"PREDICTION_SKIPPED", reason, signal_quality, data_context, explanation, recovery_authority, timestamp}`; **no `predictions` key** |
| signal gate blocked in `_check_signal_gate` | `:510-518`, saved `:594-597` | `{status:"prediction_skipped", reason, prediction_allowed:false, diagnostics, explanation, recovery_authority, timestamp}`; no `predictions` key; **status casing differs** from the row above |
| **no predictions generated** | `:631-640` | returns `{predictions:[], confidence_scores:[], metadata:{error,method}}` **without calling `_save_predictions`**, so the previous canonical file stays on disk |

**Success schema** (`:709-733`, plus `inject_agent_metadata`, `integration/metadata_writer.py:65-186`):

| field | type | source / meaning |
|---|---|---|
| `status` | str `"success"` | `:710` |
| `predictions` | **list[int]** | `:653-664`; each is `int(seq[next_idx])` from `_build_prediction_pool` (`:839-850`) |
| `raw_scores` | list[float] | model `predicted_quality` per entry (`:843`, `:648-660`) |
| `confidence_scores` | list[float] | sigmoid of the z-scored raw score, or `0.5` constant (`:694-701`) |
| `confidence_scores_normalized` | list[float] | `raw/max(raw)` or zeros (`:689-692`) |
| `metadata.method` | str | `forward_only` / `dual_sieve` (`:603-608`) |
| `metadata.pool_size` | int | **the configured pool size, not the count written** (`:717`) |
| `metadata.k`, `forward_count`, `reverse_count`, `intersection_count` | int | `:718-721` |
| `metadata.prng_type`, `mod`, `default_skip`, `dual_sieve`, `gpu_available`, `model_type`, `feature_count`, `score_stats`, `version` | — | `:722-731` |
| `metadata.timestamp` | str, **naive local time** | `datetime.now().isoformat()` (`:729`) |
| `agent_metadata.{inputs, outputs, parent_run_id, pipeline_step, pipeline_step_name, follow_up_agent, confidence, reasoning, timestamp_injected}` | — | `prediction_generator.py:929-942`; `timestamp_injected` is UTC (`metadata_writer.py:180`). `inputs` lists the model checkpoint and `survivors_with_scores.json` only, **not the draw history** |

**Example: the real host file** (`predictions/next_draw_prediction.json`, 1,483 B, mtime 2026-03-07 19:08:32 -0800,
sha256 `ee7cdd38…2a9c`; read-only copy at `data/wp2_stage_a/host_copy/`): `predictions [436, 429]`, `raw_scores
[-21.0204, -21.0205]`, `confidence_scores [0.6698, 0.3302]`, `metadata.pool_size 20`, `k 20`, `intersection_count 2`,
`model_type neural_net`, `feature_count 89`, `timestamp "2026-03-07T19:08:32.711077"`, `agent_metadata.parent_run_id
"step5_20260306_212952"`. The file says `pool_size 20` and holds 2 predictions.

**Pools of 20, 100 and 300.** Step 6 does **not** produce them. It produces one list. `_build_prediction_pool`
takes the top `pool_size` survivors by predicted quality (`:836-841`), and `generate_predictions` then keeps the
first `k` (`:653`). Defaults are `pool_size=20`, `k=10` (`:89-91`). Under WATCHER the manifest maps `--k` from
`pool_size` (`agent_manifests/prediction.json:32`), with `default_params` `k: 20, pool_size: 20` (`:181-182`), and
the CLI builder emits both (`agents/watcher_agent.py:2203-2218`). So the WATCHER-run pool is at most 20 entries.
The manifest bound is `pool_size` 5–100 (`agent_manifests/prediction.json:45-49`). The only "20/100/300" producer
on the host is `build_pools.py` (`:24-40`, `:173`, output `prediction_pools.json`). It is not Step 6, supports
only `xorshift32` (`prng_next_batch` raises for anything else, `:66-78`), and its output does not exist on the host.
Three declared-but-unproduced names also exist: `predictions/prediction_pools.json` and
`predictions/ranked_survivors.json` (gitignored `agent_manifests/definitions.json:80-84`), and
`predictions/ranked_predictions.json` (`chapter_13_orchestrator.py:877`, `test_phase_8_soak.py:60`). `ls` on the
host: all three are absent.

**Can a pool contain the same value twice? Yes.** Each ranked survivor contributes its own generator's next
value (`:841-850`), with no de-duplication. **Measured on host artifacts:** 4 of 19 archived Step 6 outputs
(`predictions_20260208_205252`, `_20260215_082503`, `_20260215_114303`, `_20260216_153308`) hold 20 entries with 19
distinct values. The other 15 are distinct. All 19 are `list[int]`. TB ruling S179 §2.5
(`docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:142-168`) already records that the generator "ranks predictions and
then truncates to `pool_size`" and does not preserve a deterministic unique ranking.

## Q2 — Everything that reads a Step 6 output

Search: `/bin/grep -rIn --exclude-dir=.git` over the whole main tree **including gitignored and untracked
files** (the shell `grep` wrapper honours `.gitignore`, so `/bin/grep` was used). Patterns: `next_draw_prediction`,
`prediction_pool`, `ranked_predictions`, `confidence_map.json`, `prediction_pools.json`. I also read `docs/` hits
by count, `systemctl list-unit-files`, `systemctl --user list-unit-files`, `crontab -l`, `/etc/systemd/system` and
`~/.config/systemd` (content grep), and `.py`/`.sh` under `/home/michael` depth ≤ 3 outside both projects.
**No systemd unit, user unit or cron entry references any of these files** (one project unit, `daily3scraper.service`,
disabled; `crontab -l` → "no crontab for michael"). Logs and copies in `backups/`, `*.bak*` and other worktrees are
non-executing and excluded below.

| reader | file:line | path it expects | schema it expects | executes in production? |
|---|---|---|---|---|
| Chapter 13 diagnostics | `chapter_13_diagnostics.py:50`, `:100-120`, `:242-265`, `:305-311`, `:774-775`; CLI `:1017`, `:1047` | `prediction_pool.json` (CWD) | `{"predictions":[{"value"…,"confidence","rank"}], "pool_size", "generated_at"}` (`:104-112`); also accepts a bare list (`:117-118`) | yes, via orchestrator `chapter_13_orchestrator.py:282` (no arguments) |
| Chapter 13 confidence map | `chapter_13_diagnostics.py:51`, `:123-129`, `:310-311` | `confidence_map.json` (CWD) | `{"<value>": float}` | yes; nothing writes it |
| Chapter 13 root cause | `chapter_13_orchestrator.py:876-937`, called `:328-330` | `predictions/ranked_predictions.json` | list, or dict with `predictions`/`ranked` + optional `draw_id`; every sampled entry needs `features` (`:924`), and also `hit`, `seed`, `rank` (`:691`, `:699`, `:710-711`) | gated by `_detect_hit_regression`, which never opens on real diagnostics (`:547-557`) |
| WATCHER results loader | `agents/watcher_agent.py:2279-2288`, called `:2255` | `prediction_pool.json`, then fallback `results_dir/step6_*.json`, `*_results.json`, `results.json` | any JSON dict | yes, after a successful Step 6; finds nothing at the root (absent) |
| WATCHER output validation | `agent_manifests/prediction.json:12`, `:20-22`, `:40-42`; `evaluation_type: file_exists` `:197` | `predictions/next_draw_prediction.json` | existence (+ size); key table `agents/watcher_agent.py:181-182` matches `predictions*.json`/`prediction_pool*.json` by filename, and `next_draw_prediction.json` matches neither [reading only; validator not executed] | yes |
| WATCHER freshness skip | log evidence e.g. `logs/s120_step5_nn.log:58` ("Step 6: Fresh: …next_draw_prediction.json … skipping") | canonical path, mtime | mtime | yes [code site not opened this session] |
| `evaluate_pools.py` | `:17-26`, `:28-51`, `:55-58` | `--pools prediction_pools.json` (build_pools format) | `{"pools":{"20":{"numbers":[…],"coverage"}}}`; truth is `daily3.json` matched by `--date`/`--session` | manual CLI only |
| `test_phase_8_soak.py` | `:60`, `:293`, `:746` | `predictions/ranked_predictions.json` | as root-cause reader | test |
| Pipeline step context (descriptive) | `agents/pipeline/pipeline_step_context.py:148` | `prediction_pool.json`, `prediction_analysis.json` | declaration only | imported (`agents/pipeline/__init__.py:13`); not a file reader |
| Declarations / prompt text | `schemas/results_schema_v1.json:928`; `bundle_factory.py:134`, `agents/contexts/bundle_factory.py:134`; `agent_contexts/step6_prediction.md:88-90`; `agent_manifests/definitions.json:80-84` (gitignored); patch installers `install_watcher_phase1_patch*.sh:130/135`; superseded gitignored manifest `step6_restoration/agent_manifests/prediction.json:14-15`, `:36` | various | none executed | no |
| Root-level `watcher_agent.py:127`, `:1322` | — | `prediction_pool.json` | — | stale copy beside canonical `agents/watcher_agent.py`; not on the executable path [not traced further] |

**Writers:** only `prediction_generator.py:896-952` writes a Step 6 output. No file anywhere writes
`prediction_pool.json`, `confidence_map.json`, `ranked_predictions.json` or `predictions/prediction_pools.json`.
This repeats the 2026-11-11 host addendum (`docs/CHAPTER_13_AUDIT.md` A.0 #1, A.7), re-checked this session on the
tree, the gitignored files, units, cron and home scripts. **Nothing reads `predictions/next_draw_prediction.json`'s
content** except WATCHER's existence/size validation. Chapter 13 never reads it.

## Q3 — Beta's control, reproduced

Harness: `data/wp2_stage_a/q3_controls.py`. It AST-extracts the **unchanged** `load_json_safe` (`:70-78`),
`load_predictions` (`:100-120`), `load_latest_draw` (`:132-169`), `compute_prediction_validation` (`:204-279`) and
`compute_confidence_calibration` (`:286-365`) from `chapter_13_diagnostics.py`, plus `evaluate_pools` (`:28-51`)
from `evaluate_pools.py`, and execs them in an isolated namespace. **No pipeline module was imported.** It ran
with `PYTHONDONTWRITEBYTECODE=1 python3 -I` and printed `COMPLETION SENTINEL: q3_controls ran to end, 18 cases`.
Log: `data/wp2_stage_a/q3_controls.log`. Results: `q3_controls_results.json`.

| case | input | result |
|---|---|---|
| (a) positive control | `[{"value":436,"rank":1},{"value":101,…},{"value":777,…}]`, draw 436 | **RETURNED** `exact_hits 1, best_rank 1, pool_size 3` |
| (b) integer format | `[436, 101, 777]`, draw 436 | **RAISED** `AttributeError: 'int' object has no attribute 'get'` |
| (c) real host canonical file, through unchanged `load_predictions` | read-only copy of `predictions/next_draw_prediction.json` (sha256 identical to source; size and mtime identical before and after the copy) | **RAISED** `AttributeError: 'int' object has no attribute 'get'` |
| (c′) all archived Step 6 outputs | 19 read-only copies of `predictions/history/*.json` | **19/19 RAISED** `AttributeError` |

So Beta's finding holds on the real artifact. If Chapter 13 were pointed at Step 6's actual output today
(`--predictions predictions/next_draw_prediction.json`, `:1017`), `generate_diagnostics` would raise at `:811`.
The CLI maps that to exit 99 (`:1079-1083`), and the orchestrator logs a cycle `outcome: "error"`
(`chapter_13_orchestrator.py:443-446`). [The CLI and orchestrator consequences are from reading; only the grading
function was executed.]

## Q4 — Draw identity

**What Step 6 records about the target draw: nothing that identifies it.**
- No date, session, draw id, history length, dataset path, dataset hash or data cutoff appears in any result
  shape (`prediction_generator.py:709-733`, `:929-942`).
- The only times are `metadata.timestamp` (naive local, `:729`) and `agent_metadata.timestamp_injected` (UTC,
  `integration/metadata_writer.py:180`). Both are **generation** times.
- Step 6 reads the history as `--lottery-history` (`:1052-1056`). For a list of dicts it keeps only `d['draw']`,
  dropping `date` and `session`. Under WATCHER the argument is `daily3.json` (`agent_manifests/prediction.json:180`),
  rewritten to the frozen absolute immutable version path (`agents/watcher_agent.py:1925-1939`). On the host that
  is `daily3-20260801T145551443433Z-513648160d35.json`: 18,068 records, last `2026-02-26 midday`, sha256
  `51364816…68f6` (`daily3_current.json`). **That identity reaches Step 6's argv and is then discarded.**
- The dataset is the combined-session container (first `2000-01-01 evening`, last `2026-02-26 midday`). Step 6
  forecasts index `next_idx = len(lottery_history)` of that combined integer stream (`:839`, `:846-849`). The
  output never states which date or session that index is meant to be. Mapping a combined-container index to a
  (date, session) is a **design decision, not a fact** the code supports. TB rulings make combined-container order
  carry no PRNG-advance meaning (fact file §2.10b). **Owner decision, not adjudicated here.**

**What Chapter 13 uses to choose the outcome.** `load_latest_draw` takes `history.get("draws")[-1]`
(`chapter_13_diagnostics.py:139-145`) from `lottery_history.json` (`:52`, orchestrator default `:282`). It
fabricates `draw_id = f"draw_{len(draws)}"` when absent (`:163`) and fills a missing `timestamp` with **now()**
(`:166`, observed in Q3 fixtures). On the host, `lottery_history.json` is `{draws, metadata}` with 5,148 draws:
5,000 `synthetic_bootstrap` and 148 `synthetic`, last `SYNTHETIC-2026-02-15-424`. **It shares no records or writer
with `daily3.json`** (also `docs/CHAPTER_13_AUDIT.md` B2, re-measured this session). Fed the pipeline dataset's
shape (a bare list), `load_latest_draw` **raises** `AttributeError: 'list' object has no attribute 'get'`
(Q3 harness, `Q5_daily3_list_shape_history`).

**Can the two be matched today? No.** The pool carries no target key, and the grader takes the last record of a
different file. The only existing draw-identity check is `load_predictions_from_disk(expected_draw_id=…)`
(`chapter_13_orchestrator.py:903-910`). It compares a `draw_id` the pool does not have against a `draw_id` from
`lottery_history.json`, and it serves only the never-opened root-cause path. `evaluate_pools.py` does match by
`(date, session)` against `daily3.json` (`:17-26`), but its pool format carries no target either. The operator
supplies `--date`/`--session` by hand (`:57-58`).

**If a new draw is appended after a pool is made, Chapter 13 grades the newest draw.** It always takes
`draws[-1]` (`:145`). Fixture `Q4_appended_after_pool_selects`: the history `[436 (T), 812 (T+1)]` returns `T+1`.
Nothing compares the pool's generation time, or anything else, with the draw. If two draws arrive between cycles,
the earlier one is never graded.

## Q5 — Missing, stale and malformed inputs

All "graded" rows below were executed in the Q3/Q5 harness on the unchanged functions unless marked [reading].

| input condition | what happens today | anchor |
|---|---|---|
| pool file missing | **silent default** `{"predictions":[], "pool_size":0}` → graded `exact_hits 0`, `pool_coverage 0.0` → **a miss**: `NO_EXACT_HITS`, `current_hit_rate 0`, `consecutive_misses+1` | `chapter_13_diagnostics.py:114`, `:73-74`; `:223-234`; `:529-547`, `:606` |
| pool is a Step 6 `PREDICTION_SKIPPED` abstention | no `predictions` key → `.get("predictions", [])` → **graded as a miss** (`q5_skip_case.log`) | `prediction_generator.py:333-341`; `chapter_13_diagnostics.py:775` |
| Step 6 produced zero predictions | canonical file **not rewritten**, so an older pool stays at the canonical path | `prediction_generator.py:631-640` |
| pool from an earlier run (stale) | no check of age, run id, target or history version; graded against `draws[-1]` | `chapter_13_diagnostics.py:774-781` (no comparison exists) |
| pool in Step 6's real format (list of ints) | **error** `AttributeError` | `:243` (Q3 b, c) |
| pool file not valid JSON | **error** `JSONDecodeError` | `:77-78` |
| prediction value a string | **error** `TypeError` | `:246` |
| prediction value outside 0–999 | **silently accepted** and counted as an ordinary miss | `:246-255` |
| confidence map missing | silent `{}` | `:129` |
| history missing | **error** `FileNotFoundError` (no default given) | `:139`, `:75` |
| history empty | **error** `ValueError` | `:142-143` |
| history as the pipeline's list shape | **error** `AttributeError` | `:141` |
| last history record has no draw value | `raw_value None` → **error** `TypeError` in grading | `:148-160`, `:246` |
| last history record has no timestamp | **silent** `timestamp = now()` written into the diagnostics as `draw_timestamp` | `:166`, `:857` |
| last record has no draw_id | **silent** `draw_{n}` | `:163` |
| model sidecar missing (root path) | silent `{"model_type":"unknown",…}` | `:184-190` (also `CHAPTER_13_AUDIT.md` B19) |
| previous diagnostics missing | first-run defaults, not an error | `:193-197` |
| orchestrator, any exception | cycle `outcome "error"`, logged to `chapter13_cycle_history.jsonl`; not graded | `chapter_13_orchestrator.py:443-446`, `:462-465` [reading] |

## Q6 — History: has a pool ever been graded against a real draw?

**Result: UNAVAILABLE as a universal claim.** On the surfaces below, **no record exists of a Step 6 pool graded
against a real draw.** Three record sets exist that look like grading but are not.

| directory / surface searched | finding |
|---|---|
| `diagnostics_history/` (233 files) | **FOUND, not real:** 233/233 have `prediction_validation.pool_size == 0`. `draw_source`: 230 `synthetic`, 3 `synthetic_bootstrap`, **0 real**. Dates by filename: 3 in 2026-01, 230 in 2026-02, last `20260215_084818`. 0/233 carry `hit_at_20`. Re-tallied this session; matches `CHAPTER_13_AUDIT.md` A.0 #3 |
| `post_draw_diagnostics.json`, `.previous_diagnostics.json` | 2026-02-15, draw `SYNTHETIC-2026-02-15-424`, `pool_size 0`, `consecutive_misses 233` |
| `chapter13_cycle_history.jsonl` (513,723 B, mtime 2026-02-15) | cycle log; not re-tallied this session (231-cycle tally in `CHAPTER_13_AUDIT.md` A.4) |
| `prng_analysis.db` → `step1_trial_history` | **0 of 56 rows** have non-null `downstream_score` or `hit_at_20`. Read per addendum §A, below |
| `backtest_temp/` (12 `metrics*.json`, mtimes 2025-10-06/07) | **FOUND, not a Step 6 pool:** `evaluate_pools.py` output against **real** `daily3.json` draws (2020-02-15, 2024-12-01…07, 2025-09-06/07, all midday). Every file has `prediction_metadata.survivor_count 0`, `prng_type xorshift32`, all three pools `hit: false`. Empty `build_pools.py` pools (`:191-208`) graded as misses. Predates the repository (first commit 2025-11-29) and Step 6 v6 |
| `watcher_kpi_*_findings.json` (5 files, 2026-07-20/21) | closed-form / Monte-Carlo analyses at the uniform null (`method: "closed-form / deterministic"`). **They grade no pool** |
| `strategy_history/`, `strategy_recommendation.json`, `diagnostics_outputs/regression_trigger_s87.json`, `shadow_*/shadow_metrics.json` | hit-field names present (`/bin/grep -lE` hit); not opened in content. **Not established either way** |
| `predictions/` | Step 6 outputs only; no grade fields |
| systemd (system + user), cron (`michael`), `/etc/systemd/system`, `~/.config/systemd` | no grading job |
| `/home/michael` depth ≤ 3 outside both projects (`.py`/`.sh`) | one backup `~/s167_safety_backup_20260424_170757/watcher_agent.py` (reader table copy); other worktrees are git copies |

**`prng_analysis.db` read (addendum §A):** no writer process was running (re-checked). Before opening:
`prng_analysis.db` 106,496 B, mtime 2026-08-21 21:29:03.138555356 -0700; `prng_analysis.db-wal`, `-shm`, `-journal`
**all absent**. Opened as `file:/home/michael/distributed_prng_analysis/prng_analysis.db?mode=ro&immutable=1` with
plain SQL (no `DistributedPRNGDatabase` constructed). After the read, a directory listing of `prng_analysis.db*`
was byte-identical to the before-listing (`data/wp2_stage_a/db_dir_before.txt` vs `db_dir_after.txt`, `diff` empty).
**Nothing was created beside the original.**

UNAVAILABLE surfaces for this question are listed in §U.

## Q7 — Other consumers of the grade

Search: `/bin/grep -rn --include=*.py` over the main tree (excluding `backups/`, `docs/`, `apply_s*`, `verify_s*`,
`*.bak`). Patterns: `exact_hits|prediction_validation|best_rank|pool_coverage|current_hit_rate|consecutive_misses|hit_values|median_rank|near_hits`,
plus `get_best_step1_params|downstream_score`. Each hit was read.

Inside `chapter_13_diagnostics.py`, `exact_hits` produces `current_hit_rate = exact_hits / pool_size` (`:529-531`),
drives `consecutive_misses` (`exact_hits == 0`, `:544-547`), and drives the flags `NO_EXACT_HITS` and `WEAK_SIGNAL`
(`:606-614`). The `--status` view reads it too (`:947`).

| consumer | file:line | reads | fields it needs |
|---|---|---|---|
| retrain triggers | `chapter_13_triggers.py:238-319`, input `post_draw_diagnostics.json` (`:53`) | `pipeline_health.consecutive_misses`, `current_hit_rate`, `window_decay`, `survivor_churn`; `confidence_calibration.predicted_vs_actual_correlation`; `summary_flags` | hit count (via both health fields), confidence |
| LLM advisor prompt | `chapter_13_llm_advisor.py:139-151`, `:258-291` | **`exact_hits`**, `pool_size`, `best_rank`, `pool_coverage`, `consecutive_misses` | hit count, hit rank, pool size |
| proposal schema summary | `llm_proposal_schema.py:250-300` | **`exact_hits`**, `pool_size`, `consecutive_misses` | hit count, pool size |
| acceptance escalation | `chapter_13_acceptance.py:476` | `pipeline_health.consecutive_misses` | miss streak |
| S140b write-back | `chapter_13_orchestrator.py:305-316` | `best_rank` → `hit_at_20/100/300` (`best_rank ≤ 20/100/300`), `pool_coverage` → `downstream_score = 0.5·h20 + 0.3·h100 + 0.15·h300 + 0.05·cov` | hit rank, coverage |
| ↳ DB | `database_system.py:484-500` | writes `step1_trial_history.hit_at_*`, `downstream_score` for `run_id = step1_{prng_type}_{seed_start}` | — |
| ↳ Step 1 warm start | `database_system.py:502-514` (`ORDER BY COALESCE(downstream_score, trial_score) DESC`), called from `window_optimizer_integration_final.py:2091` inside `if n_parallel > 1` (`:2080`) | best Step 1 params | **the grade can steer Step 1 parameter selection** on that path |
| root-cause gate | `chapter_13_orchestrator.py:547-557` | `summary_flags` containing "hit_rate"+"drop", or top-level `hit_at_20`/`previous_hit_at_20` | neither is ever produced (`CHAPTER_13_AUDIT.md` B15 Blocker 0) |
| KPI analyzers (documentation-grade) | `watcher_kpi_metricC_deterministic*.py` (e.g. `_v2_2.py:253-259`), `watcher_kpi_baseline.py`, `watcher_kpi_resolvability_probe.py` | the **definition** of `exact_hits`/`current_hit_rate` | they state explicitly that their verdicts hold "ONLY under the distinct-prediction assumption (exact_hits in {0,1} per draw)", which Q1 shows real pools violate (4/19) |

**Measured semantics of the grade fields, unchanged code (Q3 harness):**
- **Duplicates inflate `exact_hits`.** `[436, 436, 5]` vs draw 436 → `exact_hits 2`, `hit_values [436, 436]`. With
  that input, `current_hit_rate = 2/3`. Rank falls back to `predictions.index(pred)+1` (`:244`), which returns the
  first equal element's position for duplicates.
- **`best_rank` counts near hits.** `[{"value":439,"rank":1}]` vs 436 → `exact_hits 0`, `near_hits 1`,
  `best_rank 1` (`:252-258`). So the S140b write-back would record `hit_at_20 = 1.0` for a pool with **no exact
  hit** (also `CHAPTER_13_AUDIT.md` N1). With a ≤20-entry pool the three tiers are one event
  (`TB_RULING_S177_KPI_GOVERNANCE.md:303-327`).
- **`pool_coverage` is distinct-count/1000** (`:262-266`), while `pool_size` and `current_hit_rate` use the
  raw entry count (`:221`, `:531`). The numerator and denominator of the one rate in use follow different
  duplicate rules.
- **Valid draw `000` works today.** `{"draw": 0}` → digits `[0,0,0]`, `raw_value 0`. A prediction `0` → `exact_hits 1`.
- `evaluate_pools.py` computes its baseline from the pool **label**: `k = int(pool_size)` (`:38`). A `"20"`
  pool holding 2 numbers reports `random_baseline 0.02`, `lift 50.0` on a hit, where `k/1000` with distinct `k`
  would be 0.002.

## Q8 — Proposed contract. Not implemented.

The contract below follows the rulings already in force rather than inventing new ones: S177 Blocker 5
(`TB_RULING_S177_KPI_GOVERNANCE.md:300-353`), S179 §2.5 ranked-unique artifact, §2.6 Hit@K unavailability, §6
evaluation identity, §8 step 4 "Implement ranked unique prediction artifact" (`TB_RULING_S179_IMPLEMENTATION_AUTH.md:142-181`,
`:376-430`, `:513-547`), Beta's realignment §3–§4 (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:79-136`)
and addendum §E/§F. S179 is approved but **not implemented**: `git grep -nE "unique_rank|hitK_available|evaluation_id|prediction_artifact_fingerprint"`
over tracked `*.py *.sh *.json` returns no relevant hit, `git log --all -S unique_rank -- '*.py'` is empty, and the
one untracked `.py` in the main tree (`fix_coordinator_retry_logic.py`) does not match.

### 8.1 One schema (draft `step6_prediction/v2`)

```json
{
  "schema": "step6_prediction/v2",
  "status": "success | prediction_skipped",
  "forecast_id": "<sha256 of canonical JSON of the frozen body below, excluding forecast_id>",
  "target_draw": {"date": "YYYY-MM-DD", "session": "midday|evening", "draw_key": "YYYY-MM-DD/midday"},
  "data_cutoff": {
    "dataset_manifest": "daily3_current.json", "dataset_version_id": "...", "dataset_sha256": "...",
    "record_count": 18068, "last_record": {"date": "...", "session": "..."},
    "history_index_forecast": 18068, "sessions_used": ["midday", "evening"]
  },
  "configuration": {
    "parent_run_id": "step5_...", "model_checkpoint_sha256": "...", "sidecar_sha256": "...",
    "window_config_sha256": "...", "prediction_config": {"pool_size": 20, "k": 20, "...": "..."},
    "source_commit": "...", "source_tree_dirty": false, "dirty_paths": [], "code_fingerprint": "..."
  },
  "generated_at_utc": "...Z",
  "ranked": [
    {"raw_rank": 1, "unique_rank": 1, "value": 436, "quality": -21.02, "confidence": 0.67,
     "survivor_id": "<seed>", "skip": 0, "source_index": 17, "duplicate_of": null}
  ],
  "pool": {"k_requested": 20, "k_distinct": 19, "values": [436, "..."]},
  "abstention": null
}
```

Rules:
- `value` is an integer in `[0, 999]`. Anything else fails the writer before the file is written.
- `ranked` holds **every** ranked entry, with ordering `quality desc → survivor_id asc → source_index asc`
  (S179 §2.5). The first occurrence of a value takes the next `unique_rank`. Later occurrences carry
  `unique_rank: null` and `duplicate_of: <raw_rank>`.
- `pool.values` is the distinct values in `unique_rank` order, truncated to `k_requested`. **`k` = `k_distinct`**,
  the number of distinct values (Beta realignment §4). Hit@K is available only when `k_distinct ≥ K`; otherwise
  `hitK_available=false`, `hitK=null` (S179 §2.6).
- `status: prediction_skipped` carries `abstention: {reason, signal_quality}` and **no** `pool`. It is an
  abstention, never a miss.
- **Owner decision required:** how `target_draw` is chosen. Step 6 today forecasts index `len(history)` of a
  combined-session stream (Q4). The contract needs a required, explicit `(date, session)` that the producer
  **asserts**, not one inferred after the fact. Whether that means per-session pools, or a combined pool tagged
  with the next session in some declared order, is a design question for Michael. This report does not decide it.
- The write is atomic (temp file + `os.replace`), and the archive copy is byte-identical to the canonical file.
  Zero-prediction runs write an explicit `status` (`no_predictions`), so the canonical path can never silently
  keep an older pool (Q5).

### 8.2 Grading rules (any option that grades)

1. **Missing, stale or malformed input is an error, never a miss.** The grade states are `GRADED_HIT`,
   `GRADED_MISS`, `PENDING` (target outcome not yet in the dataset), `ABSTAINED` (`prediction_skipped`), and
   `ERROR_<reason>` (missing artifact, malformed JSON, schema mismatch, value out of range or wrong type, empty
   `pool` with `status: success`, target not matched). Only the first two enter the hit-rate numerator and
   denominator. `ABSTAINED` and `ERROR` are counted in their own denominators (addendum §F).
2. **The outcome is looked up by `target_draw.draw_key`** in the dataset authority: the immutable version named by
   the pointer manifest at grading time, whose `predecessor_sha256` chain must include `data_cutoff.dataset_sha256`.
   It is never `draws[-1]`, and never `lottery_history.json` unless that file is made the same authority (B2).
3. **Stale:** a forecast whose `data_cutoff.last_record` is not the record immediately before `target_draw` in
   the declared session order is `ERROR_STALE_TARGET`. A forecast whose `generated_at_utc` is later than the
   target outcome's publication is `ERROR_NOT_FROZEN`. [Publication time is not in `daily3.json` today. Until it
   is, "frozen" can only be shown by the forecast existing before the dataset version that first contains the
   target: `published_utc` in the pointer manifest.]
4. **Membership, one per target draw:** `membership_hit = (outcome ∈ pool.values)`. Hit@K uses `unique_rank ≤ K`
   over `ranked`. Per-draw baseline `k_distinct/1000`. The aggregate is
   `Σ membership_hit / N_graded` against `Σ (k_distinct_i/1000) / N_graded`, where `N_graded` counts
   `GRADED_HIT + GRADED_MISS` only.
5. **Existing fields are not reinterpreted** (addendum §E.1). `exact_hits` keeps its current per-entry meaning,
   and new fields `membership_hit`, `k_distinct`, `hit_unique_rank` and `hitK{20,100,300}{,_available}` are added
   beside it. Consumers of `exact_hits` (Q7) are then migrated one at a time. Separately: near hits must not feed
   `best_rank`-derived tier hits. That is a behaviour change to the S140b write-back, and it is called out under
   Option B.
6. **Forecast eligibility** (addendum §E.2). A forecast is eligible only if its `forecast_id` verifies, its
   `configuration` is complete, and it was frozen before its target. The rule for several eligible artifacts
   targeting the same `draw_key` must not depend on the outcome. Proposed: **the earliest frozen artifact
   (`generated_at_utc`, then `forecast_id` as the tie-break) is graded; later ones are recorded `SUPERSEDED`, not
   graded.** "Latest before cutoff" is the alternative. The choice belongs to Michael and must be declared before
   any grading.
7. **Idempotent grading:** `evaluation_id = sha256(draw_key, forecast_id, dataset_sha256, evaluator_schema_version,
   evaluator_code_fingerprint, source_commit)` (S179 §6). Repeated grading of one draw with the same
   `evaluation_id` adds nothing. A different `evaluation_id` for an already-graded `(draw_key, forecast_id)` is
   recorded and flagged, never summed.

### 8.3 Where the change lands — three options

All three keep `predictions/next_draw_prediction.json` as the canonical path. That is WATCHER's validated output
(`agent_manifests/prediction.json:12`, `:20-22`, `:40-42`) and the Beta-ruled single canonical output (`c6e3989`,
per `CHAPTER_13_AUDIT.md` C1). None of them needs an edit to `agents/watcher_agent.py` or `agent_manifests/`.

**Option A — Step 6 side only.**
- **Files edited:** `prediction_generator.py` only (`generate_predictions` `:629-738`, `_build_prediction_pool`
  `:836-862`, `_save_predictions` `:896-952`). It writes schema v2 to the canonical path. It **also** writes a
  compatibility file in Chapter 13's documented object format at `prediction_pool.json`, with
  `predictions = [{"value","confidence","rank"}]` from `pool.values` (distinct, so `exact_hits ∈ {0,1}`).
- **What it fixes:** path, schema, duplicates (by producing distinct values), and the identity fields in the
  artifact.
- **What it cannot fix:** §8.2 rules 1–3 and 6–7. Chapter 13 still treats a missing file as a miss
  (`chapter_13_diagnostics.py:114`), still grades `lottery_history.json[-1]` (`:145`), still feeds near hits into
  `best_rank` (`:252-258`), and cannot tell `PENDING` from a miss. **Option A meets the contract only partially.**
- **Readers affected (from Q2):** Chapter 13 diagnostics (begins grading real pools against the wrong draw),
  triggers / LLM advisor / acceptance / S140b → DB → Step 1 warm start (all start receiving non-empty grades),
  WATCHER `_find_results(6)` (`agents/watcher_agent.py:2286-2297` would now load `prediction_pool.json` as Step 6's
  results dict, a behaviour change **in a no-touch file without editing it**, so its effect must be checked by
  reading `:2255` onward before Stage B; **DEFERRED** for anything that would need a WATCHER edit), and WATCHER's
  key table `prediction_pool*.json → ["predictions"]` (`:182`) would start to apply. Unaffected: `evaluate_pools.py`
  and the root-cause reader (it needs `features`).

**Option B — Step 6 and Chapter 13.**
- **Files edited:** `prediction_generator.py` (as in A, **without** the compatibility file);
  `chapter_13_diagnostics.py` (loader for schema v2 at the canonical path; outcome by `draw_key` from the dataset
  authority; grade states; new membership fields beside the legacy ones; missing/stale/malformed → `ERROR_*`, not
  an empty pool); `chapter_13_orchestrator.py` (pass the canonical path explicitly instead of relying on defaults,
  `:282`; S140b tiers from `hit_unique_rank` and `hitK_available`, with no near hits; no write-back unless
  `GRADED_*`).
- **Readers affected:** `chapter_13_triggers.py` (`:238-319`: an `ERROR`/`PENDING`/`ABSTAINED` cycle must not
  advance `consecutive_misses` or fire `HIT_RATE_COLLAPSE`; S176/S179 already rule these triggers audit-only);
  `chapter_13_llm_advisor.py:258-291` and `llm_proposal_schema.py:282-300` (read `exact_hits`, kept unchanged in
  meaning); `chapter_13_acceptance.py:476`; `database_system.py:484-514` and its Step 1 consumer
  `window_optimizer_integration_final.py:2091` (a governed-file question: `window_optimizer*.py` is no-touch, and
  that consumer is not edited, but it starts receiving real `downstream_score` values, which **changes Step 1
  behaviour on the `n_parallel > 1` path**, so it needs its own ruling); the KPI analyzers (definition changes
  from per-entry to membership if they are pointed at the new field). The draw-ingestion side (B2:
  `lottery_history.json` vs `daily3.json`, `synthetic_draw_injector.py`, `draw_ingestion_daemon.py`) is a
  **prerequisite or a parallel item**, not part of this option.
- Option B is the only option that makes the live loop itself grade correctly. It is also the largest, and it
  touches the trigger semantics S179 governs.

**Option C — Step 6 plus a standalone frozen-output grader (Chapter 13 untouched).**
- **Files edited:** `prediction_generator.py` (as in B). **New** file, e.g. `tools/grade_frozen_forecasts.py`: it
  reads the archive `predictions/history/*.json` (schema v2), resolves outcomes by `draw_key` from the immutable
  dataset versions, applies §8.2 in full, and writes an append-only per-draw grade ledger (S179 §2.7 locking) under
  `data/`. That is Beta's "external frozen-output grader [that] could measure the existing box without waiting for
  all Chapter 13 automation repairs" (realignment §4).
- **Readers affected:** none of the live consumers. Chapter 13 keeps reading `prediction_pool.json`, which still
  does not exist, so the live loop keeps grading empty pools exactly as today, and that stays a documented,
  separate defect (`CHAPTER_13_AUDIT.md` B1). WATCHER validation is unchanged because the canonical path is
  unchanged.
- Smallest blast radius. The yardstick number can be measured without changing any trigger, write-back or Step 1
  input. It leaves the live loop wrong, by declaration.

**Alpha-side observation (not a decision):** A cannot meet §8.2. B and C both meet it. C changes no live
consumer, and B is the eventual integration. A sequence of C first, then B later behind S179, is possible. The
choice is Michael's.

### 8.4 Controls Stage B must pass

Positive and negative controls, each run against the **unchanged** current code first, recording the "before"
result as Q3/Q5 did, and then against the change:

| # | control | expected after change |
|---|---|---|
| P1 | object/v2 pool with one known hit (Q3 a) | `GRADED_HIT`, `membership_hit true`, `hit_unique_rank 1` |
| P2 | the same pool in v1 integer format (Q3 b) and the real host file (Q3 c) | explicit `ERROR_SCHEMA` (or a declared, tested v1 adapter), **never** an `AttributeError` traceback and never a miss |
| P3 | valid draw `000` with prediction `0` | `GRADED_HIT` |
| N1 | duplicate winning value (`[436, 436, 5]`) | `membership_hit true` counted **once**, `k_distinct 2`, baseline `0.002`; `exact_hits` keeps its legacy value only if the legacy field is retained, and is labelled |
| N2 | near hit only (`439` vs `436`) | `GRADED_MISS`; tiers `hit@20/100/300 = 0` |
| N3 | out-of-range (`1436`, `-1`) and wrong-type (`"436"`, `4.5`, `null`) values | writer refuses; grader `ERROR_VALUE` |
| N4 | missing artifact | `ERROR_MISSING`; `consecutive_misses` unchanged |
| N5 | malformed artifact (bad JSON; v2 missing `target_draw`; `forecast_id` mismatch) | `ERROR_MALFORMED` / `ERROR_FORECAST_ID` |
| N6 | `prediction_skipped` artifact (`q5_skip_case`) | `ABSTAINED`, in its own denominator |
| N7 | zero-prediction Step 6 run | canonical file rewritten with `status: no_predictions`; the previous pool is **not** left in place |
| N8 | stale target (cutoff two records before the target) | `ERROR_STALE_TARGET` |
| N9 | future target (outcome not yet in the dataset) | `PENDING`, **not** a miss, no streak change |
| N10 | history appended past the target (Q4 fixture) | grades the **target** record, not `draws[-1]` |
| N11 | repeated grading of one draw (same `evaluation_id`) | no second count; a different `evaluation_id` → flagged, not summed |
| N12 | two eligible forecasts for one `draw_key` | the declared outcome-independent rule picks one; the other is `SUPERSEDED`; result identical under a swapped outcome value |
| N13 | fewer than K distinct values | `hitK_available false`, `hitK null` (S179 §2.6) |
| N14 | target record lacking a draw value; history in list shape | `ERROR_OUTCOME`, never `now()` timestamps or tracebacks |
| N15 | uniform-random pool control over a fixed set of real draws | membership rate within a **declared** tolerance of `Σk/1000` (addendum §F), not exact equality |

Evidence caveat: all Stage A executions were CPU-only extractions of pure functions. They are **not** evidence about
a Step 6 run, about GPU execution, or about WATCHER behaviour.

---

## U. Everything UNAVAILABLE

- **Whether a pool was ever graded against a real draw on any surface other than those in Q6:** `.127` bare-metal
  fallback (not booted; Zeus runs one OS at a time), rig CT100s (no rig access by rule), ser8 and pre-repository
  archives, the private remote, root's and other users' crontabs (`/var/spool/cron/crontabs` not readable as
  `michael`), Michael's own manual runs not leaving files on this host.
- **Content of** `strategy_history/*.json`, `strategy_recommendation.json`, `diagnostics_outputs/regression_trigger_s87.json`,
  `shadow_gpu_*/shadow_metrics.json`: they match hit-field names, but were not opened in this session.
- **`chapter13_cycle_history.jsonl`:** not re-tallied this session (prior tally `CHAPTER_13_AUDIT.md` A.4).
- **WATCHER Step 6 runtime behaviour:** the argv derivation (`--k 20`, `--lottery-history <immutable path>`), the
  output-validation key-table match, and the freshness-skip code site were established by reading, not by
  execution (no WATCHER run, by rule).
- **The target session Step 6 intends for `next_idx = len(history)`:** not stated anywhere in the code read. It is
  an owner design question, not a searchable fact.
- **Draw publication times:** absent from `daily3.json`, so "frozen before outcome" cannot be shown from the
  dataset alone today.
- **Root-level `watcher_agent.py`:** treated as a stale non-executing copy. Its execution path was not traced.

## C. Commands used (host VM 101, user `michael`; Python as `PYTHONDONTWRITEBYTECODE=1 python3 -I`)

```
date -Is; pgrep -af "range_miner_coordinator|watcher_agent|window_optimizer"; ss -ltn | grep 5700
git -C /home/michael/distributed_prng_analysis status --porcelain; git worktree list; git rev-parse HEAD
git worktree add --detach /home/michael/tfm_worktrees/wp2_stage_a HEAD; mkdir -p …/wp2_stage_a/data/wp2_stage_a
ps -eo pid,args | /bin/grep -E "range_miner_coordinator|watcher_agent|window_optimizer" | /bin/grep -v grep   (x4)
git diff --stat 26d3147 HEAD -- prediction_generator.py chapter_13_diagnostics.py chapter_13_orchestrator.py agents/watcher_agent.py
Read: prediction_generator.py (all), chapter_13_diagnostics.py (all), chapter_13_orchestrator.py (all),
      integration/metadata_writer.py:65-186, evaluate_pools.py (all), build_pools.py (all),
      agents/watcher_agent.py:170-200, :480-505, :1915-1965, :2150-2225, :2240-2330,
      agent_manifests/prediction.json:1-60, :160-200, chapter_13_triggers.py:50-56, :230-320,
      database_system.py:46, :484-518, window_optimizer_integration_final.py:2080-2100,
      agents/pipeline/pipeline_step_context.py:140-152, test_phase_8_soak.py:55-64,
      install_watcher_phase1_patch_v1.1.2.sh:125-140, agent_contexts/step6_prediction.md:84-94,
      agent_manifests/definitions.json:78-86 (gitignored)
Governance/docs read: CLAUDE.md (both), the addendum, the brief, docs/CHAPTER_13_AUDIT.md (all),
      docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:20-30, :640-650,
      docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:79-159,
      docs/TB_RULING_S177_KPI_GOVERNANCE.md:296-353, docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:142-200, :376-432, :513-560,
      tfm-project-facts skill
/bin/grep -rIn --exclude-dir=.git -e <pattern> .   for next_draw_prediction, prediction_pool, ranked_predictions,
      confidence_map.json, post_draw_diagnostics, prediction_pools.json (with and without docs/ and logs/)
/bin/grep -rn --include=*.py -E "exact_hits|prediction_validation|best_rank|pool_coverage|current_hit_rate|consecutive_misses|hit_values|median_rank|near_hits" .
/bin/grep -rn --include=*.py "get_best_step1_params\|downstream_score" .
git grep -nE "unique_rank|hitK_available|evaluation_id|prediction_artifact_fingerprint|target_draw" -- '*.py' '*.sh' '*.json'
git log --all --oneline -S unique_rank -- '*.py'; git ls-files -o --exclude-standard -- '*.py' '*.sh'
git check-ignore -v predictions/next_draw_prediction.json lottery_history.json diagnostics_history/x.json
ls -la --time-style=full-iso predictions/ predictions/history backtest_temp; stat of root candidate files
find . (pruned) -name '*.json|*.jsonl|*.csv|*.tsv' -size -50M | xargs /bin/grep -lE '"(hit_20|hit_at_20|hit_100|exact_hits|hit_any_pool|downstream_score)"'
find . (pruned) -iname '*metrics*.json|*kpi*|*evaluat*|*hit_rate*|*graded*'
systemctl list-unit-files; systemctl is-enabled daily3scraper.service; systemctl --user list-unit-files; crontab -l
/bin/grep -rlE … /etc/systemd/system ~/.config/systemd; find /home/michael -maxdepth 3 (pruned) -name '*.py|*.sh' | xargs /bin/grep -lE …
cp -p predictions/next_draw_prediction.json predictions/history/*.json → data/wp2_stage_a/host_copy/ (stat before/after, sha256)
python3: read-only summaries of predictions/history, lottery_history.json, daily3.json, daily3_current.json,
      diagnostics_history/*, post_draw_diagnostics.json, backtest_temp metrics, watcher_kpi_*_findings.json
python3: sqlite3 'file:…/prng_analysis.db?mode=ro&immutable=1' (sidecars stat'd absent before; dir listing diffed after)
python3 data/wp2_stage_a/q3_controls.py <worktree> <scratch>   → q3_controls.log, q3_controls_results.json
python3 data/wp2_stage_a/q5_skip_case.py <worktree> <scratch>  → q5_skip_case.log
```

**Write record:** this report and `data/wp2_stage_a/**` (untracked scratch) inside
`/home/michael/tfm_worktrees/wp2_stage_a`. `git worktree add` also wrote the worktree's administrative entry in the
main repository's `.git/worktrees/`, as the brief's prescribed command does. **Nothing in the main working tree was
created or changed** (`git status --porcelain` empty at 19:07:24). No branch, tag or commit was created. Nothing was
pushed. No file under `predictions/`, and no `prediction_pool.json`, `confidence_map.json`, `lottery_history.json`
or diagnostics file, was written.
