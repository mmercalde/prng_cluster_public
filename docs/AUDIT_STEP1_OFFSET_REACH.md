# AUDIT — Step 1 `offset` bounds and the reachable draw-index range

**Type:** read-only investigation. **No fix authorized, none made. Nothing committed.**
**Host:** VM 101 `zeus-ubuntu-vm`, user `michael`, `/home/michael/distributed_prng_analysis`
**Tree:** HEAD `2b0d2dc`, `git status --porcelain` **empty** (clean) at audit start.
**Date:** 2026-08-12
**Falsifiable question:** *Was Step 1's `offset` parameter bounded such that the sieve only ever
examined a small slice at the front of the 18,068-draw history, rather than searching across it?*

---

## VERDICT

**CONFIRMED, and it is already recorded — in five places, one of them a RANK-1 open item.**

The binding constraint is not `offset` alone; it is the **pair** `offset ≤ 100` **and**
`window_size ≤ 50`. Together they cap the highest reachable index in the session-filtered draw
array at **149**. Every window the production sieve can position lies inside
**`data[0:150]`** of a **date-ascending, oldest-first** file.

**Measured on the live dataset this session: 150 of 18,068 records = 0.830%.**
Union across all three session options: **300 distinct records = 1.660%.**

Two precisions on the report as received:

1. **`offset` is not "set to 100."** It is an Optuna search *dimension* sampled per trial over
   `[0, 100]` (`trial.suggest_int('offset', …)`). `100` is the **upper bound of the bound**, which
   is what makes the reach a ceiling rather than a single position.
2. **`skip` does not widen the slice.** It widens reach in **PRNG-output space only**. Proven at
   the kernel level in Q4 — the per-draw loop is bounded by `k = window_size` in both the constant
   and the hybrid kernels. So the slice is genuinely ~150 draws wide, not wider.

Michael's recollection is accurate in substance. The value **100** is real, it is in
`distributed_config.json`, and it does confine the search to the front of the history.

---

## 1. What are `offset`'s actual bounds, and where are they set?

**Live bounds: `min = 0`, `max = 100`.** Single source of truth, with one hardcoded fallback and
one dataclass default that agree with it.

| # | role | anchor | value |
|---|---|---|---|
| 1 | **authoritative config** | `distributed_config.json` → `search_bounds.offset` (**lines 80–83**) | `{"min": 0, "max": 100}` |
| 2 | fallback dict if config missing/corrupt | `window_optimizer.py:74` | `"offset": {"min": 0, "max": 100}` |
| 3 | `SearchBounds` dataclass defaults | `window_optimizer.py:142-143` | `min_offset: int = 0` / `max_offset: int = 100` |
| 4 | config → dataclass load | `window_optimizer.py:165-166` | `min_offset=cfg["offset"]["min"]`, `max_offset=cfg["offset"]["max"]` |

**The two Optuna sampling sites** — both bounded by the same object:

- `window_optimizer_bayesian.py:532-534` — `trial.suggest_int('offset', bounds.min_offset, bounds.max_offset)`
- `window_optimizer_integration_final.py:2236-2238` — same call, `_local_bounds`

> ⚠ **Anchor drift, recorded for correction:** the skill (§2.7 instance 5, §2.20) cites the
> sampling site as `window_optimizer_bayesian.py:423`. **Live it is `:532-534`.** The behaviour is
> unchanged; only the line number is stale.

**Non-Optuna paths, all capped at the same 100:**
`window_optimizer.py:231` (`random.randint(self.min_offset, self.max_offset)`), `:407`
(`self.offsets = offsets or [0, 100]`, grid strategy), `window_optimizer_bayesian.py:1027-1030`
(evolutionary, `np.clip` to the same bounds), `:1103`, and
`window_optimizer_integration_final.py:2779` (`offsets=[0, 100]`).

**There is no `--offset` CLI argument on Step 1.** The only operator-facing offset input is
`--warm-start-offset` (`window_optimizer.py:1516-1517`), which reaches
`study.enqueue_trial(_ws_params)` (`window_optimizer_bayesian.py:786`); the objective still calls
`suggest_int` against the same `[0, 100]` distribution. *(I did not empirically test enqueueing an
out-of-range value — the warm-start values in production originate from prior trial history, which
was itself produced inside `[0, 100]`.)*

### 1.1 The bound has no derivation — and one patch proves it by contrast

`distributed_config.json`'s `window_size` block carries **two** rationale keys
(`_calibration_note` S148, `_s172_note` S172). **`offset` carries no `_note` of any kind.**

`apply_s139_window_max_50.py` is the decisive artifact. Its docstring reasons explicitly about
`window_size` (500→50, *"167-trial Optuna run confirmed short-term temporal regime"*) and
`skip_max` (500→250). In the very same dict it rewrites, it passes `offset` through
**byte-identical** — `apply_s139_window_max_50.py:63` (old) and `:72` (new), both
`"offset": {"min": 0, "max": 100}`. Every other entry in that dict came out carrying an inline
`# S139:` comment. **Offset came out carrying nothing.**

**Origin in git history:** `search_bounds.offset = {min:0, max:100}` enters the repo at
**`dfcba45`** *("Session 8: Centralize search bounds + fix threshold optimization")*, with no
accompanying note. It has never been modified since. The initial commit `0101306` has an
operational `"offset": 0` value but no `search_bounds` block.

### 1.2 A fourth, inert declaration says 2000 — do not read it as the live bound

`agent_manifests/window_optimizer.json` → `parameter_bounds.offset` declares **`"max": 2000`**
(lines 87–95). **It is inert.** Per the three-hop parameter route (skill §2.15), hop 1 is
WATCHER's declared-key filter (`agents/watcher_agent.py`, `if key in declared`), and `offset` is
**absent from `default_params`** and **absent from `args_map`** — verified live by loading the
manifest this session. Nothing reads that `2000`. Its description,
*"Time offset from current draw position,"* is also wrong (see Q2).

---

## 2. What does `offset` index INTO? Trace to the kernel argument.

**`offset` has TWO simultaneous meanings driven by ONE payload scalar.** This is Chapter 2's
finding **F-4**, `CONFIRMED, not repaired`.

### 2.1 Role A — head-relative index into the SESSION-FILTERED draw array

```python
# miner/range_miner_worker.py
:642   if sessions:
:643       data = [e for e in data if e.get("session") in sessions]
:644   n = len(data)
:649   start = max(0, min(int(offset), n - window_size))
:650   window = data[start:start + window_size]
```

Note the ordering: **the session filter runs first, so `offset` indexes the filtered list, not raw
file positions.** This matters for Q3 and is the one place the published summaries are least
explicit.

The identical clamp appears in **nine** loaders (`/bin/grep` across the tree):
`miner/range_miner_worker.py:649` · `window_optimizer_integration_final.py:266` ·
`sieve_gpu_worker.py:113` · `sieve_filter.py:184` · `reverse_sieve_filter.py:114` ·
`sieve_filter_INTEGRATED.py:79` · `reverse_sieve_filter_INTEGRATED.py:57` ·
`reverse_sieve_filter_TEST_ORIGINAL.py:57` · `tests/phase6/known_answer_reference.py:161`.
*(The `_INTEGRATED` / `_TEST_ORIGINAL` copies are the known stale duplicates — not proposing any
action on them.)*

It is `data[start : start+window_size]` — **a slice from index 0, i.e. from the OLDEST end**,
because the file is date-ascending (`daily3.json[0]` = `2000-01-01 evening`, measured this
session). This is pinned by a content gate: `tests/test_chapter2_content_gate.py:576` asserts the
literal string `"min(int(offset), n - window_size)"` is present in the source.

### 2.2 Role B — device pre-advance count, i.e. a kernel argument

The same scalar rides to the GPU. Full chain, miner path:

```
payload["offset"]                              miner/range_miner_worker.py:875
  -> BuildContext(offset=offset, ...)          miner/range_miner_worker.py:948
  -> builder(ctx) -> _offset_tail(ctx)         miner/range_miner_worker.py:951, :197-198
  -> ScalarArg(ctx.offset, "int32")            miner/range_miner_worker.py:198
  -> materialize_kernel_args -> _gpu_launch    miner/range_miner_worker.py:952, :956
  -> kernel formal parameter `int offset`      prng_registry.py:964
```

And in the kernel body — `java_lcg_flexible_sieve`, the TFM production family:

```c
prng_registry.py:964    unsigned long long a, unsigned long long c, int offset
prng_registry.py:973        unsigned long long state = seed & m;
prng_registry.py:974        for (int o = 0; o < offset; o++) {
prng_registry.py:975            state = (a * state + c) & m;      // <-- pre-advance the generator
prng_registry.py:976        }
```

The same `for (int o = 0; o < offset; o++)` pre-advance appears in ~20 kernels across
`prng_registry.py` (`:424`, `:481`, `:533`, `:645`, `:694`, `:1102`, `:1236`, `:1392`, `:1668`,
`:1824`, `:1888`, `:2395`, `:2464`, `:2548`, `:2615`, `:2707`, `:2772`, `:2851`, …).

**Independent parallel route:** the same value also reaches the residue derivation on the
coordinator/host side — `window_optimizer_integration_final.py:266` uses `config.offset` for the
identical slice, and `miner/range_miner_coordinator.py:6640` / `:8664` place `"offset": offset`
into the stripe-assign payload. One value, two consumers, no separation.

### 2.3 The exception — forward hybrids never receive it

`build_java_lcg` (`miner/range_miner_worker.py:215-236`): in the **hybrid forward** branch the
tail is `a, c` only — `:220` states it in the source, *"forward: uint64 a, c — ABI-critical, NO
offset"*. `_hybrid_prefix` (`:178-193`) carries **13 elements, none of which is `offset`**. The
file header (`:30-35`) confirms four of six hybrid families take **NO offset**.

**So on a forward-hybrid trial, `offset` still selects the host window slice but never reaches the
kernel at all.** That is skill §2.7 instance 5, chain row *"Optuna `offset` → forward hybrid ✅ ✅
✅ ✗ — dies in kernel args"* (behaviour model `:982`).

### 2.4 Four incompatible written definitions (all four verified live this session)

| # | source | definition | live status |
|---|---|---|---|
| 1 | `window_optimizer.py:106` docstring | *"Time offset from current draw"* | **wrong** — head-relative, from the oldest end |
| 2 | `agent_manifests/window_optimizer.json:92` | *"Time offset from current draw position"* | **wrong**, and inert (§1.2) |
| 3 | `config_manifests/parameter_registry.json:38-43` | *"advance seeds by `offset*(skip+1)` before testing"*, `cli_flag: --offset` | **not what any loader does**; no such CLI flag exists on Step 1 |
| 4 | the loaders + the kernels | head-relative array slice **and** device pre-advance | **what actually runs** |

This is the DIVERGENT register entry **D11** (`docs/PIPELINE_BEHAVIOUR_MODEL.md:1187`), which
records all four and Beta's disposition.

---

## 3. Given `max_offset = 100` and the `window_size` cap, what can the sieve reach?

### 3.1 The arithmetic

With `n` (filtered) ≫ 150, `min(offset, n − window_size)` is always `offset`, so `start = offset`.

```
start          = offset            ∈ [0, 100]
window         = data[start : start + window_size],  window_size ∈ [6, 50]
highest index  = start + window_size − 1  ≤  100 + 50 − 1  =  149
```

Union over every legal `(offset, window_size)` pair = filtered indices **`{0 … 149}` = exactly 150
records**. A **single trial** sees at most `window_size ≤ 50` consecutive draws; 150 is the
envelope of everything reachable across all trials, all runs, ever.

`window_size` bounds verified live: `distributed_config.json` → `search_bounds.window_size`
`{min: 6, max: 50}`.

### 3.2 Measured against the live dataset

Dataset measured this session: `daily3.json`, **18,068 records**, `2000-01-01 evening` →
`2026-02-26 midday`. Confirmed against the authoritative pointer manifest `daily3_current.json`
(`record_count: 18068`, `sha256 513648160d35…`, lineage `daily3-combined-L001`).

Because `offset` indexes the **session-filtered** array (§2.1) and `sessions` is itself an Optuna
dimension (`window_optimizer_bayesian.py:535`, three options at `window_optimizer.py:181-186`),
the answer differs per session option:

| `sessions` | filtered n | reachable filtered idx | **reachable raw idx** | reachable dates | **fraction of 18,068** |
|---|---|---|---|---|---|
| `['midday','evening']` | 18,068 | 0 … 149 | 0 … 149 | 2000-01-01 … **2000-05-29** | **150/18,068 = 0.830%** |
| `['evening']` | 9,553 | 0 … 149 | 0 … 149 | 2000-01-01 … **2000-05-29** | **150/18,068 = 0.830%** |
| `['midday']` | 8,515 | 0 … 149 | **1,039 … 1,337** | 2002-11-04 … **2003-04-02** | **150/18,068 = 0.830%** |

**Union across all three session options: 300 distinct records = 1.660% of the file**, spanning
raw indices 0 … 1,337, i.e. `2000-01-01` … `2003-04-02`.

Two measured facts worth carrying:

- **`['both']` and `['evening']` reach the byte-identical set of records.** Raw indices 0–149 are
  *all evening records* — the CA midday draw did not exist yet. First midday record in the file is
  **raw index 1,039, `2002-11-04`** (measured). So the "both sessions" option buys no additional
  coverage at the front of the file.
- **`['midday']` reaches a strictly disjoint, later block** — it is the only session option that
  reaches past 2000, and it still stops in **April 2003**.

### 3.3 Corroboration from a real production config

`optimal_window_config.s162_victory.json` (held in the repo root) reconstructs the best documented
production result — `W6_O64_evening_S3-37`, *"887 bidirectional survivors, 42:36 elapsed, 26/26
GPUs stable."* Resolving that config against the live dataset this session, the **six draws it
actually sieved** were:

```
2000-03-05 · 2000-03-06 · 2000-03-07 · 2000-03-08 · 2000-03-09 · 2000-03-10   (all evening)
```

`docs/DAILY3_CONSUMER_CONTRACT_v1.md:198-208` performs the same resolution for the then-live
`optimal_window_config.json` (`window_size: 21`, `offset: 66`, both sessions) and gets
`data[66:87]` = `2000-03-07 evening … 2000-03-27 evening`. I re-derived that slice against the
current file and it reproduces exactly.

> **Live-state note (§1.2 discipline):** `optimal_window_config.json` **does not currently exist**
> in the repo root — only `optimal_window_config.s162_victory.json` and
> `optimal_window_config_test.json` (the latter carries `window_size: 454`, far outside
> `[6,50]`; it is a test artifact, not a production config). The consumer contract's worked
> example describes a prior artifact, not the present tree.

> **Anchor-verification note.** Every `file:line` in this report was re-checked against the live
> file after drafting. Six anchors were off by one or two lines in the first pass and were
> corrected (`range_miner_worker.py` session filter, `k = len(residues)`, the arg-materialize
> pair; `prng_registry.py` constant-kernel match/stride/rate lines; the hybrid `search_min`
> block; `enqueue_trial`). The anchors in this document are the corrected ones.

---

## 4. Does `skip` extend that reach in draw-index space, or only in PRNG-output space?

**PRNG-output space ONLY. The slice is genuinely ~150 draws.** Proven at the kernel level, for
both skip modes.

### 4.1 Constant skip — `java_lcg_flexible_sieve`

```c
prng_registry.py:972    for (int skip = skip_min; skip <= skip_max; skip++) {
prng_registry.py:973        unsigned long long state = seed & m;
prng_registry.py:974-976    for (int o = 0; o < offset; o++)  state = (a*state + c) & m;   // pre-advance
prng_registry.py:977-979    for (int s = 0; s < skip;   s++)  state = (a*state + c) & m;   // stride
prng_registry.py:981        for (int i = 0; i < k; i++) {                 // <<< k = window_size
prng_registry.py:982            state = (a * state + c) & m;
prng_registry.py:984-986        if (((output % 1000) == residues[i] % 1000) && …) matches++;
prng_registry.py:987-989        for (int s = 0; s < skip; s++) state = (a*state + c) & m;  // stride
prng_registry.py:991        float rate = ((float)matches) / ((float)k);
```

The draw loop is `i < k`, and `k = len(residues)` (`miner/range_miner_worker.py:879`) — the length
of the host slice. `skip` burns generator outputs **between** residues; it never advances `i`
beyond `k−1` and never reaches a residue outside `data[start : start+window_size]`.

### 4.2 Variable skip (hybrid) — `java_lcg_hybrid_multi_strategy_sieve`

```c
prng_registry.py:1029   for (int draw_idx = 0; draw_idx < k && draw_idx < 2048; draw_idx++) {
prng_registry.py:1033-1034   int search_min = expected_skip − skip_tolerance;   // clamped at 0
                             int search_max = expected_skip + skip_tolerance;
prng_registry.py:1035        for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
prng_registry.py:1042-1044       if (… == residues[draw_idx] % 1000 …) { matches++; expected_skip = test_skip; …
prng_registry.py:1060   float match_rate = (float)matches / k;
```

Same conclusion, and stronger: the greedy per-draw adaptive search re-centres `expected_skip` on
each hit, so it explores a *wide* region of PRNG-output space — but `draw_idx` is still bounded by
`k`, and the array it indexes is still the same `window_size`-long slice.

### 4.3 The quantitative gap

Under the live bounds, a constant-skip trial can consume up to
`offset + skip_max × (window_size + 1)` ≈ `100 + 250 × 51` ≈ **12,850 generator outputs** — while
covering at most **50 observed draws**, all inside `data[0:150]`.

**That is the whole point of skip** (skill §0.4): it models unpublished outputs — pre-test draws,
other games drawn in the same session, power cycles. It is a hypothesis about *what happened
between the draws you can see*. It is structurally incapable of changing *which* draws you can
see. Nothing in the search space presently does that except `offset`, and `offset` is capped
at 100.

---

## 5. Is this recorded anywhere as a known defect, ruling, or open item?

**YES — recorded in five places, with a live RANK-1 disposition. It is a STATUS, not a new
finding.** Reporting it as a discovery would be the governance error §1.1 warns about.

| # | source | what it records | status |
|---|---|---|---|
| 1 | **`docs/DAILY3_CONSUMER_CONTRACT_v1.md:198-212`** (§4.3, *"`offset` slices from index 0 — i.e. from the **oldest** end"*) | The mechanism and its consequence in as many words: **"The production sieve analyses draws from March 2000."** Worked example `data[66:87]`. Also flags `parameter_registry.json:38-43` as stale. | **GOVERNING CONTRACT, live.** Also carried at `:488-489` in its own findings table. |
| 2 | **`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_FROM_PROCEDURES.md:602-640`** (§C.3, headed *"Where the search cannot reach the data the document governs — **the finding**"*) | The full reach table per session filter, against the first CA-procedures-governed record. Verdict in the document's own words: **"THE PRODUCTION SIEVE CANNOT EXAMINE A SINGLE DRAW THAT THIS DOCUMENT GOVERNS."** Explicitly self-labels *"Governed status, not a new finding."* | **OPEN.** |
| 3 | **`…ATTACK_PLAN_FROM_PROCEDURES.md:744-790`** (§D.1) | **RANK 1 of the whole report: "Move the analysis window onto draws the document governs."** *"C.3 shows the current geometry cannot reach a single governed draw. Until this changes, no approach in this report is actually testing the document… This is the prerequisite, not a preference."* Names the required change: `search_bounds.offset.max`, currently `100`. | **OPEN — the highest-ranked outstanding analytical item found in this audit.** |
| 4 | **`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1133`, `:1346`, §7.3** — finding **F-4**; and **`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:318-348`, `:2257`** — conflict **C-2**; registered as **D11** at `docs/PIPELINE_BEHAVIOUR_MODEL.md:1187` | The dual-role defect that makes the bound un-raisable by config edit: one scalar drives host slice **and** device pre-advance, *"coherent only at `skip = 0`."* | **CONFIRMED, NOT REPAIRED.** Beta ruling (`CHAPTER_1:348`): settles C-2 as an **OBSERVED INCONSISTENCY, not the repair**; belongs in the future **hybrid input-semantics design**, **not a standalone arithmetic patch**. |
| 5 | **`docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:65-68`, `:561-567`, `:747`** | *"The single most decision-relevant fact this pass turned up… the sieve window is anchored at the OLDEST end of the dataset and cannot presently reach recent draws."* Row 6 of its divergence table: *"The reachable window is the oldest ~0.8% of the dataset, and `offset.max` carries no `_note` and no in-repo rationale."* | Recorded, deliberately **not** raised as a defect (that brief forbade proposing). |

Plus the tracked skill copy, **`docs/TFM_PROJECT_FACTS_SKILL.md:1106-1118` §2.21 "THE 150-DRAW
CONFOUND"**, committed at **`c7058d8`** (skill v19). Its consequence clause is the one that
matters operationally:

> *Selectivity spread, feature importance, survivor counts, S112's W8 result, S107's flat-signal
> finding — all were measured on 2000-2003 draws… **No conclusion drawn from historical trial data
> can distinguish "property of the metric/system" from "property of that window."***

### 5.1 What is NOT recorded

Two genuine absences, stated as such after the searches in §6:

1. **No TB ruling, ruling request, or proposal anywhere in the governance trail sets, justifies,
   or revisits `offset.max = 100`.** I read the full `TB_RULING_*` / `TB_RULING_REQUEST_*` /
   `PROPOSAL_*` / `TEAM_ALPHA_*` filename inventory and grepped all of `docs/` for `offset`. The
   bound enters at `dfcba45` with no note, survives `apply_s139` untouched while its neighbours
   are re-derived, and is never revisited. **The value 100 has no derivation on any surface I
   searched.**
2. **The reach is not in `docs/BACKLOG.md`.** The only `offset` line there (`:164`) is the
   unrelated `full_scoring_worker.py:305` holdout-offset item. So the RANK-1 item at source 3
   above is **not tracked in the maintained register** — it lives only inside a report.

---

## 6. Surface enumeration (VIR-6)

**Audit claim scope:** the bounds, semantics and reachable draw-index range of Step 1's `offset`
parameter at HEAD `2b0d2dc`, and whether that reach is recorded on any project surface.

### Searched

| surface | how |
|---|---|
| **Governance trail** (`TB_RULING_*`, `TB_RULING_REQUEST_*`, `PROPOSAL_*`, `TEAM_ALPHA_*`, `CLAUDE_CODE_*`) | full filename inventory + `/bin/grep -rni 'offset' docs/ --include='*.md'`; hits **read**, not counted |
| **Chapters** | `CHAPTER_1_WINDOW_OPTIMIZER.md` (§3.1.2, C-2, `:2255-2257`), `CHAPTER_1_AUDIT_v1.md` (C-2 `:385`), `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (F-4 `:1133`, `:1346`, §7.3) |
| **`docs/PIPELINE_BEHAVIOUR_MODEL.md` §16 DIVERGENT register** | **D11 found and read** (`:1187`); also `:841-842`, `:982` |
| **`docs/PROJECT_FILE_CATALOG.md`** | grepped for `offset` — two hits, both unrelated (`:100`, `:422`) |
| **`docs/BACKLOG.md`** | grepped — one unrelated hit (`:164`) |
| **SESSION_CHANGELOG corpus** | covered by the recursive `docs/` grep |
| **`apply_s*.py` / `verify_s*.py` patch corpus** | `/bin/grep -ln` across all; `apply_s139_window_max_50.py` **read in full** |
| **Code** | `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`, `miner/range_miner_worker.py`, `miner/range_miner_coordinator.py`, `prng_registry.py` (kernel bodies read, not just signatures), `sieve_gpu_worker.py`, `sieve_filter.py`, `reverse_sieve_filter.py`, `tests/test_chapter2_content_gate.py` |
| **Config / manifests** | `distributed_config.json`, `agent_manifests/window_optimizer.json` (**loaded via `json`**, not just grepped) |
| **Gitignored files** | `git check-ignore -v` run. **`daily3.json` (`.gitignore:41 *.json`) read and measured**; **`config_manifests/parameter_registry.json` (same rule) read at `:38-43`** — both invisible to any clone-based audit |
| **Git history incl. deleted** | `git log -S'"offset"' -- distributed_config.json` → origin `dfcba45`; `git log --diff-filter=D --name-only --all` for deleted offset-related files (`check_offset.py` recovered — **empty at its last revision**, no content) |
| **Host state (VM 101 live filesystem)** | `optimal_window_config*.json` inventory; `optimal_window_config.s162_victory.json` read; live dataset resolved and measured with Python |

Tool note: the shell `grep` here is a ugrep wrapper honouring `.gitignore` (which ignores `*.json`).
**All JSON searches in this audit used `/bin/grep` or a Python `json.load`.**

### NOT searched — declared

- **Pre-repository archives on ser8.** The project predates its repository (initial commit
  2025-11-29). If a rationale for `100` was written before that, it would live there. **This is
  the one surface that could still falsify §5.1's absence claim**, and I did not reach it.
- **The CA draw-procedures PDF** — not in the repo (known open item, skill §0.4).
- **Rig host state** (`.122`/`.156`/`.164`) — not relevant; `offset` is resolved coordinator-side
  and travels in the stripe payload.
- **`archives/cleanup_20251130_073217/backups/*.backup_offset_*`** (4 files, Oct 2025) — surfaced
  in the deleted-file sweep, **not opened**. They are `coordinator.py` / `timestamp_search.py`
  backups, a different `offset` (timestamp search), so judged out of scope — but I did not read
  them, and say so rather than imply coverage.

### Verification-integrity controls (VIR-1…6)

- **execution proof:** every anchor in this report was produced by a command run this session on
  VM 101; measurements printed with counts and dates, not asserted.
- **clean control:** `git status --porcelain` empty at start; all reads against HEAD `2b0d2dc`.
- **fault-injection control:** *not applicable* — this is a read-only trace, no detector was built.
- **completion sentinel:** each command's output was read in full; no truncated pipelines
  (`| head` used only on inventories where the full set was separately enumerated).
- **unavailable-observer behaviour:** ser8 archives are declared **UNAVAILABLE (not searched)**,
  never treated as clean. §5.1's absence claim is scoped to the surfaces listed as searched.
- **audit claim scope / searched / unavailable surfaces:** above.
- **governance trail searched:** yes — `TB_RULING*`, `TB_RULING_REQUEST*`, `PROPOSAL*`,
  `TEAM_ALPHA*`, `CLAUDE_CODE_*`. **chapters searched:** 1, 2, plus behaviour model §16.

---

## 7. Summary answers

1. **Bounds:** `[0, 100]`. `distributed_config.json` `search_bounds.offset` (lines 80–83) is
   authoritative; mirrored at `window_optimizer.py:74`, `:142-143`, loaded `:165-166`. **No
   `_note`, no derivation, no ruling — anywhere I searched.** The manifest's `max: 2000` is inert.
2. **Indexes into:** the **session-filtered, oldest-first draw array**, head-relative —
   `miner/range_miner_worker.py:649-650`. **And simultaneously** the generator pre-advance count,
   reaching the kernel as `int offset` (`prng_registry.py:964`, loop `:974-976`) via
   `_offset_tail` → `ScalarArg` (`miner/range_miner_worker.py:197-198`). One scalar, two roles —
   Chapter 2 **F-4**. Forward hybrids receive **no** `offset` at all (`:220`).
3. **Reach:** highest reachable filtered index `= 100 + 50 − 1 = 149` ⇒ **`data[0:150]`**.
   **150 of 18,068 = 0.830%** per session option; **union across all three = 300 records =
   1.660%**, `2000-01-01` … `2003-04-02`. A single trial sees ≤ 50 draws.
4. **Skip:** extends reach in **PRNG-output space only** (≈12,850 outputs at the bounds) — the
   draw loop is bounded by `k = window_size` in both the constant (`prng_registry.py:981`) and
   hybrid (`:1029`) kernels. **The slice is genuinely ~150 draws.**
5. **Recorded:** **yes, five surfaces**, incl. a **RANK-1 open item** (`ATTACK_PLAN_FROM_PROCEDURES`
   §D.1) and a Beta-ruled structural blocker (**F-4 / C-2 / D11: CONFIRMED, not repaired** —
   raising the bound is a *window-anchor / generator-phase separation*, **not a config edit**).
   **Not** recorded: any derivation for the value `100`, and the RANK-1 item is **absent from
   `docs/BACKLOG.md`**.

**No fix proposed. No file outside `docs/AUDIT_STEP1_OFFSET_REACH.md` was created or modified.**
