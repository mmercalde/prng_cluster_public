# CLAUDE CODE REPORT — IS `bidirectional_selectivity` A PER-SEED QUANTITY?

# VERDICT: **TRIAL-LEVEL.** Alpha's reading is correct.

**Date:** 2026-08-08 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:**
`/home/michael/distributed_prng_analysis` · **HEAD:** `746b545`
**Type:** read-only audit. Nothing launched, nothing edited, nothing fixed, nothing committed.
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_SELECTIVITY_PER_SEED_AUDIT.md`

`bidirectional_selectivity` is computed **once per (trial × skip-mode)** as a ratio of two set
cardinalities and stamped verbatim onto every survivor record of that trial. It carries **zero**
per-seed information and cannot discriminate between seeds by construction.

> ## ⚠ THE STRUCTURAL VERDICT IS ALREADY GOVERNED — REPORTED AS STATUS, NOT AS A DISCOVERY
>
> `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:501` (tracked at `fa6b713`) already
> classifies the field **TRIAL-AGG**, defined at `:479-480` as *"a trial-level scalar stamped
> identically onto every record of that trial+mode."* Its **O3** (`:1006-1008`) states *"18 of 22
> NPZ columns carry no per-seed information"*; its **O5** (`:1015-1018`) gives the formula. Skill
> §2.3 carries the same fact. `docs/CHAPTER_2_SOURCE_MAP_v1.md:341` and
> `docs/CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md:555` repeat it.
>
> **Per skill §1.1, re-reporting a governed fact as a finding is a governance error.** What is new
> in this report is confined to §5 and §6 and is stated there explicitly.

---

## 1. IS IT TRIAL-LEVEL OR PER-SEED? — the code says so in its own comments

`window_optimizer_integration_final.py`, constant-skip path, read live at `746b545`:

```python
:1755   # ACCUMULATE CONSTANT SKIP SURVIVORS WITH METADATA
:1756   # v3.0: Per-seed match rates stored individually, not trial aggregates
:1758   if accumulator is not None:
:1759       # Trial-level context (same for all seeds in this trial)      <<<<<<
:1760       # v3.1: Compute trial-level intersection statistics           <<<<<<
:1761       _union_size = len(forward_set | reverse_set)
:1762       metadata_base = {
   …
:1773           # Trial-level counts                                      <<<<<<
:1774           'forward_count': len(forward_records),
:1775           'reverse_count': len(reverse_records),
:1776           'bidirectional_count': len(bidirectional_constant),
:1778           'intersection_count': len(bidirectional_constant),
:1779           'intersection_ratio': len(bidirectional_constant) / max(_union_size, 1),
:1780           'forward_only_count': len(forward_set - reverse_set),
:1781           'reverse_only_count': len(reverse_set - forward_set),
:1782           'survivor_overlap_ratio': len(bidirectional_constant) / max(len(forward_set), 1),
:1783           'bidirectional_selectivity': len(forward_set) / max(len(reverse_set), 1),
:1784           'intersection_weight': len(bidirectional_constant) / max(len(forward_set) + len(reverse_set), 1),
:1785       }
```

### The attaching loop the brief asked for — quoted, `file:line`

```python
:1793       for seed in bidirectional_constant:
:1794           fwd_rate = forward_map[seed]
:1795           rev_rate = reverse_map[seed]
:1796           accumulator['bidirectional'].append({
:1797               'seed': seed,
:1798               'forward_match_rate': fwd_rate,             # v3.0: per-seed
:1799               'reverse_match_rate': rev_rate,             # v3.0: per-seed
:1800               'score': (fwd_rate + rev_rate) / 2.0,       # v3.0: per-seed avg
:1801               **metadata_base                            # <<< SAME DICT, EVERY SEED
:1802           })
```

**`metadata_base` is built once, outside the loop, and dict-splatted into every record.** The only
quantities re-read per seed are `forward_map[seed]` and `reverse_map[seed]`. The code's own
comments mark exactly three fields `# v3.0: per-seed` and the metadata block `# Trial-level`.

**Hybrid path — identical shape:** `metadata_base_hybrid` built at `:1866-1889` (selectivity at
`:1887`), splatted at `:1898-1904`.

**The miner path is the same, and it is the certifying path.** `utils/canonical_records.py`:

```python
:234       "bidirectional_selectivity": len(fwd_set) / max(len(rev_set), 1),
:237   records = []
:238   for seed in sorted(both):                   # ascending seed order (§6)
:239       fwd_rate, rev_rate = fwd_map[seed], rev_map[seed]
```

Same construction: one trial-scoped dict at `:222-236`, one per-seed loop from `:238`.

**Not recomputed per seed anywhere.** `forward_set` / `reverse_set` are the trial's full forward and
reverse survivor populations; their cardinalities do not depend on which seed is being written.

---

## 2. EMPIRICAL PROOF AGAINST HELD ARTIFACTS

Read-only (`np.load`, `allow_pickle=False`); no NPZ was modified. Five artifacts on this host
carry the field:

| artifact | seeds | **distinct `bidirectional_selectivity`** | value(s) | distinct (trial, mode) | groups with >1 value |
|---|---:|---:|---|---:|---:|
| `d6_release_grade_20260729/generation_root/…all.npz` **(certified)** | **319** | **1** | `1039.5718` | 1 | **0** |
| `d6_release_grade_20260729/d6_zeus_smoke/…all.npz` | 319 | **1** | `1039.5718` | 1 | **0** |
| `tfm_forensics/pre_d3_accumulator_20260725/…all.npz` | **20,949** | **1** | `0.0` | 1 | **0** |
| `s167_safety_backup_20260424_170757/…all.npz` | **20,916** | **1** | `0.0` | 1 | **0** |
| `harness_npz/fixtures/prior_v2_full_schema.npz` | 700 | **1** | `1.2000` | 1 | **0** |

**Every artifact: exactly one distinct value across up to 20,949 seeds. Zero (trial, mode) groups
contain more than one value.** The certified release-grade generation
(`gen-20260730T002104136270Z-step1_java_lcg_0`, skill §2.8) carries `1039.5718` on all 319 rows —
alongside `intersection_ratio`, `survivor_overlap_ratio` and `intersection_weight`, each also with
exactly 1 distinct value, while `forward_matches` has 2 and `score` has 2.

### ⚠ Limit of the empirical proof, stated plainly (VIR-6)

**No multi-trial NPZ exists on this host.** Every artifact carrying the field has exactly one
`(trial_number, skip_mode)` pair. I checked the checkpoint and flush temporaries
(`.s172_checkpoint/…/incremental_survivors_all.npz`, `*.flush.tmp.npz`, `*.ckpt.tmp.npz`) — they
carry only `seeds` + `score` (2 arrays) or the 8-field checkpoint schema, **no metadata columns**.

**So the "one value per trial" half is proven empirically; the "N trials → N values" half is proven
only structurally, from the code at §1.** The brief's alternative acceptance criterion — *"several
distinct values matching a trial count"* — could not be exercised. The first criterion — *"a single
NPZ with exactly one distinct value across thousands of seeds"* — is met four times over, twice at
>20,000 seeds.

**Note on the two forensics artifacts:** they carry `trial_number = 0` and every metadata column
zeroed, including `forward_matches`/`reverse_matches`. They predate the D3 accumulator work and
demonstrate the *shape*, not current behaviour. **The release-grade artifact is the current-behaviour
evidence.**

---

## 3. IS THE S107 MEASUREMENT CONSISTENT WITH THIS? — yes, exactly

S107 reports (`docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:32-47`): 6,739 survivors, min = p25 =
median = p75 = p90 = **1.0099**, max **2.4711**, mean **1.0222**, **98.8% at floor**, *"Only ~81
seeds have any selectivity above the floor."*

### What produces that shape under the trial-level reading

The accumulator's L2 merge keeps **exactly one record per seed**
(`utils/run_finalizer.py:714-745`, `_select_l2_winners`: *"Exactly one record per seed; the result
is INDEPENDENT of input order"*), selected by the frozen Ruling-D key
(`_l2_sort_key`, `:690-711`): highest float32 `score` → lowest `trial_number` →
constant-before-variable within a trial.

**So each surviving row carries its *winning trial's* metadata.** The distribution of
`bidirectional_selectivity` across an accumulated NPZ is therefore **the distribution of trial-level
values weighted by how many seeds each trial won** — it is a measure of **trial dominance**, not of
anything about seeds.

**98.8% at one value means one trial won ~6,658 of 6,739 rows.** The arithmetic closes:
`6739 × 0.988 ≈ 6658`, leaving **81** — exactly the *"~81 seeds"* S107 reports. And S107's own field
table gives `bidirectional_count` max = **6702** (`:58`), i.e. one trial's intersection contained
6,702 seeds against a 6,739-row total. **Consistent to within rounding.**

### Does the merge affect it?

**Yes, and in the direction that makes the degeneracy worse.** Before the merge a seed appearing in
5 trials would contribute 5 rows with 5 different selectivity values. After the merge it
contributes **one**. The merge therefore *collapses* what little inter-trial spread reaches the
final artifact, concentrating it on whichever trials won the score tiebreak. The strict `>`
comparison the brief names is the L3 array-domain prior/current comparison
(`utils/run_finalizer.py:792`, *"`>` is strict"*); the per-seed collapse is L2's, at `:745`.

---

## 4. HOW FAR DOES IT REACH? — consumer enumeration

| # | consumer | `file:line` | needs per-seed variance? |
|---|---|---|---|
| 1 | **NPZ column 16 of the frozen 22-array contract** | `utils/canonical_records.py:234`; `window_optimizer_integration_final.py:1783`, `:1887` | **No** — storage is agnostic. Legitimate as a trial annotation |
| 2 | **ML feature merge (batch)** | `survivor_scorer.py:777` (in the 18-field `for field in [...]` loop, `:774-781`) | **YES.** A feature constant within a trial contributes no within-trial gradient |
| 3 | **ML feature default-fill (batch)** | `survivor_scorer.py:460`, `:790` — `features.setdefault(k, 0.0)` | **YES** — same feature |
| 4 | **ML feature merge (sequential fallback)** | `full_scoring_worker.py:453` | **YES** — and see the P0 note below |
| 5 | **Feature registry declaration** | `config_manifests/feature_registry.json` → path **`/per_seed_features/bidirectional_sieve_metadata/bidirectional_selectivity`** | **YES by declaration** — §5 |
| 6 | **Step-2 objective, v4.1** | `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md:112-114` | **YES** — the residue filter selects seed *subsets*; a subset mean of a constant is that constant |
| 7 | **Trained-model sidecar field list** | `bidirectional_survivors_binary.meta.json:29` | inherits from #2 |
| 8 | **NPZ contract gates** (schema/dtype only) | `tests/test_prng_encoding.py:74`, `:206`; `test_s172_phase5_d3_5_finalizer.py:91`; `test_s172_phase5_d3_columnizer.py:98`; `test_s172_phase5_d3_0_encoding_contract.py:107`; `test_s172_phase5_d4_serial_backend.py:128`; `test_s172_d6_2_checkpoint_reconciliation.py:310`; `tests/phase6/wall_ab_gate.py:151`; `tests/smoke_s172_phase5_d6_zeus_single_gpu.py:102` | **No** — they assert presence, dtype and value preservation, all of which hold |

**Rows 2–6 require per-seed variance to be meaningful and cannot have it.** Rows 1, 7 and 8 are
unaffected.

**Governed status already attached to row 4:** `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:162`
records that the sequential fallback merges only 6 fields against the batch path's 18, so
`bidirectional_selectivity` is one of **seven** features that silently become `0.0` on GPU-batch
failure — **Team Beta classifies this P0.** Reported as status; it is a *different* defect on the
same field.

---

## 5. IS THERE A GOVERNANCE RECORD OF WHAT IT WAS *INTENDED* TO MEASURE?

**Yes — and the intended definition matches the code, while the intended *scope* contradicts it.**

**The one formal definition** — `config_manifests/feature_registry.json`, JSON path
`/per_seed_features/bidirectional_sieve_metadata/bidirectional_selectivity`:

```json
{ "type": "float", "range": [0.0, null], "higher_is_better": null,
  "description": "Ratio indicating bidirectional filtering strength: forward_count/reverse_count" }
```

**The description is correct and agrees with `:1783` exactly.** `forward_count`/`reverse_count` are
themselves TRIAL-AGG (trace `:495-496`). **But the entry is filed under `per_seed_features`.**

> ### ★ NEW — the registry declares a trial-level quantity to be a per-seed feature, in the same entry that correctly defines it as a ratio of trial-level counts.

**I searched for a prior record of this collision and found none.** `/bin/grep -rn "per_seed_features" docs/*.md`
returns only schema-shape mentions (`COMPLETE_OPERATING_GUIDE_v1_1.md:567`, `v2_0.md:634`,
`PROPOSAL_Unified_Agent_Context_Framework_v3_2_4.md:2173,2176`, `v3_2_5.md:2176,2179`,
`TECHNICAL_SPEC_Feature_Remediation_Phases_2_4.md:459`) — **none names this field or this conflict.**
The descriptive trace classifies the field TRIAL-AGG but does not mention `feature_registry.json`.
**No evidence found** that anyone joined the two.

**Other governance mentions, none of which define scope:**
- `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:146` — *"bidirectional_selectivity | Precision
  of intersection"*. **This description is wrong on its own terms**: the formula
  `len(fwd)/max(len(rev),1)` **contains no intersection term** — already recorded as trace **O5**
  (`:1015-1018`). Status.
- `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:501` — the correct classification.

---

## 6. DID S107 OR ANY LATER WORK IDENTIFY THIS?

### 6.1 What v4.1 gave as its rationale — the brief's §6 question

`docs/TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md:112-114`, verbatim:

> **Why `bidirectional_selectivity`:**
> From the NPZ stats: min=1.01, max=2.47, mean=1.022
> **This is NOT flat — there is real variance to optimize against.**

> ### ★ NEW — this is a category error, and it is the origin of the whole v4.1→v4.4 lineage.
>
> The spread `min=1.01 … max=2.47` is **variance between trials**, read off an accumulated
> multi-trial NPZ. The objective it justifies operates on **subsets of seeds selected by a residue
> filter** (`scorer_trial_worker.py:385-389`), which needs **variance between seeds within reach of
> that filter**. There is none, by construction. **Inter-trial spread was read as intra-trial
> variance.**

### 6.2 v4.2's stated root cause is misattributed — confirming the brief's hypothesis

`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:49-50`:

> `bidirectional_selectivity` cannot serve as the primary quality signal **for this dataset**.

**The conclusion is correct. The stated root cause is wrong.** It is not a property of *this
dataset* — it is a property of *the field's construction*. The field would be degenerate on every
dataset, for every PRNG family, at every window size, forever. Changing the dataset cannot change
it. **The brief's hypothesis is confirmed.**

### 6.3 ★ NEW — the approved *replacement* carries the same error, and it also went dead

`TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md:75-80` proposes `bidirectional_count`:

> **Semantic meaning:** how many times **this seed** appeared in the bidirectional intersection
> across all Optuna trials in Step 1. A seed that survived 6,702 intersections is far more reliable
> than one that survived only 6.

**That semantic reading is false.** `bidirectional_count` is `len(bidirectional_constant)`
(`window_optimizer_integration_final.py:1776`) / `len(both)` (`utils/canonical_records.py:228`) —
**the size of that trial's intersection set**, marked **TRIAL-AGG** at trace `:497`. It is not, and
has never been, a per-seed appearance count. The observed range 6→6702 is the range of *trial
intersection sizes*. **I searched for any prior challenge to this reading and found none** —
`/bin/grep -rn "bidirectional_count" docs/*.md | grep -i "across all\|appeared\|survival frequency\|per-seed"`
returns only the claim itself and two documents repeating it
(`CHAPTER_3_SCORER_META_OPTIMIZER.md:446`, `SESSION_CHANGELOG_20260222_S108.md:32`, both
*"survival frequency"*).

**It was deployed** — `scorer_trial_worker.py:3` (*"v4.2 - Subset-Selection, bidirectional_count
signal, TB S107"*), `:198-211`, `:254-256` (hard-fails if absent).

**And it died the same death.** `scorer_trial_worker.py:413-418`, live comment:

```python
:413    # v4.3: Enrichment objective (TB ruling S107)
:414    # bc_score (median percentile-rank) is structurally dead:
:415    # 79.2% of pool at bc>=11300 => any large subset has constant median.
:416    # Residue arithmetic has no structural correlation with bc tier.
```

**79.2% of the pool at one tier is the same degeneracy as 98.8% at one floor value, for the same
reason.** The code names the symptom (*"structurally dead"*, *"constant median"*) and gives the
distribution — **and never names the cause.**

### 6.4 What v4.3/v4.4 actually converged on — and why it works

The live objective abandoned per-seed quality signals drawn from TRIAL-AGG fields entirely
(`scorer_trial_worker.py:420-441`): **enrichment over `skip_mode`** (`:421-425`,
`log(p_subset/p_global)` for the `skip_mode == 1` minority island) plus **coverage over
`trial_number`** (`:428-430`, `uniq_sel / uniq_total`) and a size penalty.

**This is correct use of trial-level data.** `skip_mode` and `trial_number` are CATEGORICAL/CONFIG
(trace `:506`, `:491`) — using them as **grouping labels** for enrichment and coverage is valid
precisely because a label does not need within-group variance. Using one as a **per-seed quality
score** is not.

> **So the pipeline converged on the right answer empirically, twice, without the diagnosis ever
> being written down.** The trace named the cause (TRIAL-AGG) without connecting it to Step 2; the
> Step-2 lineage named the symptom four times without connecting it to the cause. **No document
> joins them. That is this report's only substantive contribution.**

### 6.5 Did Beta rule? — **RULED, both, but not as `TB_RULING_*` documents**

**There is no `TB_RULING_STEP2_*` file.** `ls docs/ | grep TB_RULING` yields only four rulings
(`S176`, `S177`, `S178`, `S179`) and seven `TB_RULING_REQUEST_*`.

**The rulings exist and are recorded elsewhere:**
- `docs/PROJECT_FILE_CATALOG.md:61` — v4.1: **"RULED → v4.1 deployed cleanly (19/19 checks).
  SUPERSEDED BY v4.2 in the same session."**
- `docs/PROJECT_FILE_CATALOG.md:62` — v4.2: **"RULED → v4.2; lineage continued to v4.3 and v4.4 in
  the same session (`S107_session_log.md`)."**
- In-code: `scorer_trial_worker.py:198` *"(TB ruling S107 Q1-Q3)"*, `:413` *"(TB ruling S107)"*,
  `:379` *"(TB Tweak 6)"*; `CHAPTER_3_SCORER_META_OPTIMIZER.md:447` *"(TB Q2)"*.
- `docs/S107_session_log.md` (2026-02-22) records the v4.4 run: 100/100 trials collected, best
  accuracy 0.3644 at trial 32.

**Answer: both were ruled on, in-session, with the rulings captured in the catalog, the session log
and code comments rather than in standalone ruling documents. Neither remains an open request.**

---

## 7. SIBLINGS — other fields whose per-seed use is inconsistent with trial-level computation

Per the brief, Chapter 2 **F-1** (`intersection_count` duplicating `bidirectional_count` is
deliberate) is **not** re-reported. Reporting only per-seed *use* inconsistency:

| field | computation | in the ML feature merge? | verdict |
|---|---|---|---|
| **`bidirectional_count`** | `len(both)` — `:1776`, `canonical_records.py:228` | **Yes** — `survivor_scorer.py:774` | **Same defect. Plus it is the v4.2 Step-2 signal (§6.3).** Highest-consequence sibling |
| `intersection_ratio` | `len(both)/max(union,1)` — `:1779`, `:230` | **Yes** — `:775` | Same defect. Also the v4.4 secondary term |
| `survivor_overlap_ratio` | `len(both)/max(len(fwd),1)` — `:1782`, `:233` | **Yes** — `:775` | Same defect |
| `intersection_weight` | `len(both)/max(len(fwd)+len(rev),1)` — `:1784`, `:235` | **Yes** — `:778` | Same defect |
| `forward_count` / `reverse_count` | `len(forward_records)` / `len(reverse_records)` — `:1774-1775` | **Yes** — `:774` | Same defect |
| `forward_only_count` / `reverse_only_count` | `len(fwd−rev)` / `len(rev−fwd)` — `:1780-1781` | **Yes** — `:779` | Same defect |
| `intersection_count` | `len(both)` | Yes — `:775` | **F-1, not re-reported** |

**All ten TRIAL-AGG columns of the 22-array contract appear in the 18-field ML merge at
`survivor_scorer.py:774-779`.** Confirmed empirically in the certified NPZ: `intersection_ratio`,
`survivor_overlap_ratio` and `intersection_weight` each have **exactly 1 distinct value** across
all 319 rows.

**The four genuinely per-seed columns** (`seeds`, `forward_matches`, `reverse_matches`, `score` —
trace `:511-512`, skill §2.3) are the only ones that can carry within-trial signal. S107's own table
gives them std 0.032 / 0.030 / 0.030 (`:65-67`) — narrow, but non-degenerate.

---

## 8. ★ CORRECTION TO MY OWN PRIOR REPORT — as the owner asked

`docs/CLAUDE_CODE_REPORT_ATTACK_PLAN_BLACKBOX_REEVAL.md` §1.2 and §3 (mine, this session) state
that the 6,739 / 98.8% figure is

> ~~"one axis of manifold composition, already measured"~~

and, in the D.3 re-ranking, that *"the measurement partly exists already."*

**Both statements are wrong, and the verdict above is why.**

`bidirectional_selectivity` is constant within a trial. After the L2 merge each seed contributes one
row carrying its winning trial's value (§3). **The distribution of that field across an accumulated
NPZ measures how many seeds each trial won in the merge — trial concentration — and says nothing
whatever about how survivors differ from one another.** "98.8% at floor" is the statement *one trial
won ~6,658 of 6,739 rows*. It is not an axis of manifold composition; **it is not a measurement of
the manifold at all.**

**The stronger correction:** my §1.2 offered that figure as evidence that the manifold's composition
had been measured. **No such measurement exists.** By construction, 18 of the 22 columns cannot
produce one. Any real measurement of manifold composition must come from `forward_matches`,
`reverse_matches` and `score` — which is exactly the point at which skill §2.3's governed finding
bites: **`forward_matches` and `reverse_matches` are absent from the Step-3 merge list**, so the two
independent per-seed signals do not reach the ML at all.

**Net effect on that report:** D.3 Stage 1 is **not** partly done. Its premise — that survivor
population structure is measurable from held artifacts — survives, but the field I cited as already
measuring it cannot, and the fields that could are the ones a governed schema decision is still
pending on. **The rest of that report's D.3 argument is unaffected; only the "already measured"
claim is withdrawn.**

---

## 9. WHAT IS NEW VERSUS WHAT IS STATUS — the summary a reviewer needs

| claim | status |
|---|---|
| The field is a trial-level set-cardinality ratio stamped per seed | **GOVERNED** — trace `:501`, `:479-480`, O3 `:1006`, O5 `:1015`; skill §2.3; source map `:341`; pipeline overview `:555` |
| 18 of 22 NPZ columns carry no per-seed information | **GOVERNED** — trace `:511-519`, O3 |
| `bidirectional_selectivity` contains no intersection term | **GOVERNED** — trace O5 |
| The sequential fallback zero-fills seven features including this one | **GOVERNED, Beta P0** — system map `:162` |
| The identical defect class was found and fixed for `forward_matches`/`reverse_matches` in Feb 2026 | **GOVERNED** — `SESSION_CHANGELOG_20260221_S103.md:12`, `:133-136` (*"trial-level count stamped on every seed from same trial … zero signal for ML ranking"*) |
| The seven trial-level intersection statistics were **knowingly** restored the next day | **GOVERNED** — `SESSION_CHANGELOG_20260222_S104.md:57` (*"All 7 are trial-level statistics (same value for all seeds from the same trial)"*) |
| **`feature_registry.json` files it under `per_seed_features`** | **★ NEW** — §5. No evidence found of any prior record |
| **v4.1's rationale read inter-trial spread as per-seed variance** | **★ NEW** — §6.1 |
| **v4.2's stated root cause ("for this dataset") is misattributed to the dataset** | **★ NEW** — §6.2. Confirms the brief's hypothesis |
| **v4.2's replacement `bidirectional_count` was described as a per-seed appearance count and is not one; it deployed and died the same death** | **★ NEW** — §6.3 |
| **The trace named the cause without reaching Step 2; the Step-2 lineage named the symptom four times without reaching the cause** | **★ NEW** — §6.4 |

**The S103/S104 pair is the sharpest context.** On 2026-02-21 the project diagnosed *"all quality
fields in the NPZ were identical for every seed from the same trial — zero signal for ML ranking"*
and fixed it for the match-rate fields. On 2026-02-22 it restored seven trial-level statistics
while explicitly documenting that they are *"the same value for all seeds from the same trial."* **On
2026-02-22 — the same day — S107 selected one of those seven as Step 2's per-seed quality signal.**
The knowledge and the error are one day apart in the same changelog series.

---

## 10. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every code anchor read live at `746b545` this session via
  `awk`-with-line-numbers / `/bin/grep -n`. NPZ figures produced by a read-only `numpy` script under
  `~/venvs/torch` (`np.load(..., allow_pickle=False)`); script retained in the session scratchpad.
  Registry path extracted by recursive JSON walk, not by grep.
- **clean control:** the same script reports `forward_matches` (2 distinct) and `score` (2 distinct)
  in the certified artifact alongside `bidirectional_selectivity` (1) — **the detector distinguishes
  varying from constant columns on the same file**, so "1 distinct" is not an artifact of the method.
- **fault-injection control:** the cross-tabulation reports *"groups with >1 distinct selectivity"*,
  which would be non-zero if any trial group carried variance. It is 0 in 5/5 artifacts; the same
  counter is non-zero-capable by construction (it counts `len(set) > 1`).
- **completion sentinel:** all seven brief items answered; verdict stated in the first line.
- **unavailable-observer behavior:** §2 declares explicitly that **no multi-trial NPZ exists on this
  host**, so half the brief's acceptance criterion could not be exercised, and marks the structural
  half as code-derived rather than measured. §5 and §6.3 state **"no evidence found"** where the
  search returned nothing, rather than inferring absence of intent.
- **audit claim scope:** the field's construction, its storage, its consumers, and the Step-2
  lineage. **No claim that any of this should be changed** — the brief forbids proposing, and I
  propose nothing.
- **searched surfaces:** live tree at `746b545` — `window_optimizer_integration_final.py`,
  `utils/canonical_records.py`, `utils/run_finalizer.py`, `survivor_scorer.py`,
  `full_scoring_worker.py`, `scorer_trial_worker.py`, `config_manifests/feature_registry.json`,
  `bidirectional_survivors_binary.meta.json`, the `tests/` corpus · **five held NPZ artifacts across
  four directories outside the repo** (`d6_release_grade_20260729`, `tfm_forensics`,
  `s167_safety_backup_20260424_170757`, `harness_npz/fixtures`) · a bounded `find` for `*.npz` over
  `/home/michael`.
- **governance trail searched (binding order applied — trail → chapters → code):**
  `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`, `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`,
  `PROJECT_FILE_CATALOG.md` §1.1 rows 61–62, `S107_session_log.md`,
  `SESSION_CHANGELOG_20260221_S103.md`, `SESSION_CHANGELOG_20260222_S104.md`,
  `SESSION_CHANGELOG_20260222_S107.md`, `SESSION_CHANGELOG_20260222_S108.md`,
  `PROPOSAL_ML_Architecture_Remediation_v2_0.md`, `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`
  — plus `ls docs/ | grep TB_RULING` to establish which rulings exist as documents.
- **chapters searched:** `STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md` §5, §5.1, O3–O5 (the
  authority) · `CHAPTER_2_SOURCE_MAP_v1.md:341` · `CHAPTER_3_SCORER_META_OPTIMIZER.md` §7.2 ·
  `CHAPTER_1_WINDOW_OPTIMIZER.md:97`, `:1854` · `CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md:555`.
- **unavailable surfaces:** the S107-era NPZ that produced the 6,739 / 98.8% statistics is **not on
  this host** — §3's reconstruction is arithmetic consistency with S107's reported numbers, **not a
  re-measurement**, and is labelled as such · any multi-trial NPZ · ser8 pre-repository archives ·
  rig-deployed source · Beta ruling texts external to the tree.
- **⚠ stale-anchor note:** the descriptive trace cites `:903` / `:1007` for the legacy path; live at
  `746b545` those are `:1783` / `:1887`. Line drift only — the trace's classification is unchanged
  and was re-verified against live source, not relayed.
- **⚠ source-reliability note:** `CHAPTER_3_SCORER_META_OPTIMIZER.md` is audited as **24 of 55 claims
  false** (skill §2.17b). Its §7.2 objective snippet is cited **only** as evidence of what was
  documented, never as evidence of live behaviour; the live objective was read directly from
  `scorer_trial_worker.py:355-441`.

---

# 11. WHAT THIS REPORT IS NOT

- **Not a fix, and not a proposal.** The brief says *"Do not fix anything you find"* and
  *"Propose nothing and change nothing."* Nothing is proposed, including for §5's registry
  misclassification and §6.3's replacement-semantics error.
- **Not a claim that the v4.1→v4.4 remediation was wrong in outcome.** v4.3/v4.4 abandoned the
  degenerate signals and use trial-level fields as grouping labels, which is valid (§6.4). The
  outcome is sound; the recorded reasoning is not.
- **Not a re-report of governed facts.** §9 separates the 6 governed claims from the 5 new ones.
  Chapter 2 F-1 is excluded per the brief.
- **Not authorization.** Beta holds gate 12 and the Phase-7 soak (skill §8). Nothing was launched.
