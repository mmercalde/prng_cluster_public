# Chapter 8 audit — PRNG registry (component chapter; no step numbering)

Repo: `/home/michael/distributed_prng_analysis` on VM 101, user `michael`. Code base = `a15b524`
(`a15b5247336255801de8cf67d52653701b1e8c80`). Current HEAD is `b6cb789`. It changed only `agents/watcher_agent.py`,
`agents/watcher_dispatch.py` and `tests/test_watcher_safety_repairs.py` (`git diff --name-only a15b524 HEAD`). This
audit cites none of those three files, so no "repaired at `b6cb789`" note applies. Every code file cited below is
identical at `a15b524` and in the working tree (`git diff --quiet a15b524 -- <file>`, checked for `prng_registry.py`,
`utils/prng_encoding.py`, `sieve_gpu_worker.py`, `sieve_filter.py`, `miner/range_miner_worker.py`, `hybrid_strategy.py`,
`survivor_scorer.py`, `full_scoring_worker.py`, `scorer_trial_worker.py`, `tests/test_prng_encoding.py`).
Audited: `docs/CHAPTER_8_PRNG_REGISTRY.md` (1,058 lines at `a15b524`; the working-tree copy was byte-identical before this pass).
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1 (§0, §1, §3, §5) and the batch-3 common brief.
Method: read-only. Governance trail first, then chapters, then code, then host state. Every anchor below was opened in
this session. The only thing executed was read-only introspection: `python3 -I -c` importing `prng_registry` and
`utils.prng_encoding` under `~/venvs/torch` to list keys, plus `git log`/`git show`, `ls`, `cat`, and one run of
`tests/test_chapter2_content_gate.py` as the clean control. **No kernel was compiled, no GPU was touched, and there
was no rig access.**
Numbering: **neither scheme**. Chapter 8 is a component chapter (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Where a
step is named, it is the EXECUTABLE scheme: Step 2 = scorer meta-optimizer, Step 3 = full scoring, and the sieve runs
inside Step 1.
Framing held throughout: TFM is functional mimicry of surface output. A registry family is a substrate for mimicry, not a
claim about the machine, and no finding here assumes a "true seed" or a "true generator".

---

## 1. Verdict

- Claims checked: **66**
- Accurate: **35**
- BROKEN: **0**
- SWITCHED-OFF-BY-RULING: **1**
- DESCRIPTION-ONLY: **11**
- UNVERIFIED: **2**
- Absorbed into an existing item (not counted separately): **17**

35 + 0 + 1 + 11 + 2 + 17 = **66**. The reconciliation table is §10.

Null results first:

1. **No new BROKEN item (0 of 66 claims).** Every registry-related break that touches this chapter is already held
   elsewhere and is cross-referenced, not re-reported:
   - Chapter 4 B4: `_cpu_batch_generate` returns unreduced values, and 42 of 44 families reach it.
   - Chapter 4 S1 / `docs/BACKLOG.md` §8.1: the CPU reference applies skip once, where the kernel applies it between draws.
   - Chapter 2 §5.4: hybrid kernels take no skip bounds.
   - Chapter 2 §7.3 / skill §2.7 #5: six forward hybrids take no `offset`.
2. **The registry's headline count is wrong, but the error is already governed.** The chapter says 46 / 14 families.
   The live registry has **44 entries, 11 bases × 4 variants** (`prng_registry.py:3729-4132`; live
   `len(KERNEL_REGISTRY)` = 44). This was recorded as `docs/PIPELINE_BEHAVIOUR_MODEL.md` D18 (`:1194`) and in the
   RANGE-MINER spec (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:86`, `:339`, `:732-733`). It is labelled
   DESCRIPTION-ONLY (D1), not new.
3. **"Complete PRNG coverage" is true of the registry and not of execution.** RANGE-MINER covers 6 of 11 bases
   (24 of 44 entries). The other 5 raise `NotImplementedError` by the approved spec (S1). PWC covers the same 6 with 7
   branches. Both figures in `CLAUDE.md` §7 were **confirmed** against code.
4. **The chapter has no account of RANGE-MINER or of the Phase-0 encoding.** Both change what "adding a PRNG" means
   (D10). New §15 and §16 were added to the chapter.

---

## 2. Brief questions and the coverage facts

| # | Question | Result | Evidence |
|---|---|---|---|
| Q1 | Family count, re-derived live | **44 entries, 11 bases, 4 variants each** (11 `{base}`, 11 `_hybrid`, 11 `_reverse`, 11 `_hybrid_reverse`) | live import; `KERNEL_REGISTRY` `prng_registry.py:3729-4132`. Bases: java_lcg, lcg32, minstd, mt19937, pcg32, philox4x32, sfc64, xorshift128, xorshift32, xorshift64, xoshiro256pp |
| Q2 | `CLAUDE.md` §7: "7 hardcoded `family_name ==` branches covering 6 base families" | **Confirmed.** Branches at `sieve_gpu_worker.py:217` xorshift32, `:221` pcg32, `:223` lcg32, `:227` java_lcg + `_reverse`, `:232` java_lcg_hybrid + `_hybrid_reverse` (inner split `:259`), `:299` minstd, `:302` xorshift128. Generic tail `:304` | `sieve_gpu_worker.py:208-306` |
| Q3 | `CLAUDE.md` §7: 5 uncovered = mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp | **Confirmed** | `COVERED_FAMILIES` `miner/range_miner_worker.py:106-108`; `UNCOVERED_FAMILIES` `:110-111`; `resolve_builder` raises `NotImplementedError` naming the family, "no kernel launched" `:474-485` |
| Q4 | Hybrid kernels have a different signature (extra `skip_sequences`; `a,c` vs `offset`) | **Confirmed**, and wider than java_lcg alone: 6 of 11 forward hybrids have no `offset` (xorshift32 `:862`, java_lcg `:1005`, minstd `:1136`, xorshift128 `:1274`, philox4x32 `:1558`, sfc64 `:2011`). All 11 reverse hybrids end in `int offset` | parse of all 44 kernel signatures; `PHASE_INCAPABLE_VARIANTS` `miner/range_miner_worker.py:159-161` lists the 4 covered ones; Chapter 2 §7.3 holds the item |
| Q5 | Shared encoding (Phase 0, `2389b61`) | registry-derived, alphabetical, 44 ids, `java_lcg` = 0, hard-fail both directions, `ENCODING_VERSION` 3.2.0, test pins 44 | `utils/prng_encoding.py:30-92`; `tests/test_prng_encoding.py:98-102`, `:236-243` |
| Q6 | §5 host question: `agent_manifests/definitions.json` | read on the host: 220 lines, gitignored (`.gitignore:41`, `*.json`), mtime 2026-01-01. Registry-relevant content: `data_context.fields` includes `prng_hypothesis` (`:40`); recovery option `change_prng_type` (`:138`); example `prng_hypothesis {prng_type: java_lcg, mod: 1000}` (`:211-213`). No family count, no family list, nothing that contradicts §1 | `cat`, `/bin/grep -n` |

---

## 3. What actually works end to end today

Each item is a complete code path on reading. None was run.

1. **Registry lookup.** `get_kernel_info` raises `ValueError` on an unknown name (`prng_registry.py:4187-4191`). Live callers include `sieve_filter.py:206`, `:381`, `:563` and the miner's variant validation (`miner/range_miner_worker.py:548`, `:628-640`).
2. **Encoding round-trip.** The NPZ `prng_type` id is derived from the registry and hard-fails in both directions (`utils/prng_encoding.py:42-92`).
3. **RANGE-MINER variant gate.** It advertises exactly the 24 validated concrete variants (`supported_variants`, `miner/range_miner_worker.py:628-640`). An uncovered family raises before any GPU work (`:474-485`). Import-time assertions keep the phase-capability sets and the arity table consistent with those 24 (`:189-200`).
4. **Host-side residue reversal for reverse variants** (`miner/range_miner_worker.py:1201`; legacy `sieve_filter.py:231-233`, `:394-396`; PWC `sieve_gpu_worker.py:188-191`).
5. **PyTorch generation for `java_lcg` / `java_lcg_hybrid`** (`prng_registry.py:95-150`). Consumed by executable Step 3 via `survivor_scorer.py:570-572`.

---

## 4. BROKEN

**None (0).** Breaks touching the registry that are already held elsewhere, and therefore not re-reported:

- Chapter 4 **B4** (`docs/CHAPTER_4_AUDIT.md` §4): the CPU fallback is not mod-reduced; 42 of 44 families.
- Chapter 2 **§5.4** (OPEN; skill §2.7 #4): hybrid kernels declare no `skip_min`/`skip_max`.
- Chapter 2 **§7.3** (OPEN; skill §2.7 #5): forward hybrids without `offset`.

One candidate was considered and rejected as BROKEN: `get_cpu_reference` raises `KeyError` for the 18 reverse
entries that have no `cpu_reference`. Every live caller found passes a base/forward name (`full_scoring_worker.py:254`,
`:291`; `synthetic_draw_injector.py:227`; `survivor_scorer.py` via `self.prng_type`). One tool already falls back to
the base entry (`generate_fingerprint_library.py:418-426`). No pipeline path fails, so this is placed as DESCRIPTION-ONLY (D5).

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — Five registered base families do not execute on the certifying engine.**
- Chapter: §14 "Total 46 — Complete PRNG coverage"; §1.1 "kernel library for the entire system"; §11 implies that a registered family is runnable.
- Code: 20 of 44 entries (mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp × 4 variants) have kernels in the registry but no
  builder in `kernel_args_builders` (`miner/range_miner_worker.py:464-471`). `resolve_builder` raises `NotImplementedError`
  (`:474-485`). `UNCOVERED_FAMILIES` is at `:110-111`.
- Held by: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (Status APPROVED, `:18`). §11.I (`:586-588`): "must fail fast with
  `NotImplementedError` until a dedicated kernel-args builder is implemented and tested. No silent fallthrough; no degraded
  execution path." §5.3 (`:370-372`): "Coverage for each uncovered family is a separate per-family session, prioritised by owner."
  Under owner ruling §0.1, this is a review position that the owner may keep or lift.
- Lifting it would enable: one builder per family, each implemented and tested in its own session, plus its `EXPECTED_KERNEL_ARITY`
  row and phase-capability classification. With those, the certifying engine could sieve that family and emit survivors under the
  22-array NPZ contract, and the family would be on hand as a further mimicry substrate.
- Not found in any committed audit (`/bin/grep -n -i "uncovered\|COVERED_FAMILIES" docs/CHAPTER_*AUDIT*.md`: no hit), so it is counted here.

---

## 6. DESCRIPTION-ONLY

| # | Chapter says | Code at `a15b524` | Anchor |
|---|---|---|---|
| D1 | §1.1 "46 PRNG implementations across 14 families" | 44 entries, 11 bases. *GOVERNED as `PIPELINE_BEHAVIOUR_MODEL.md` D18* | `prng_registry.py:3729-4132`; live import |
| D2 | header "Lines: 4,323" | 4,235 (4,322 at `0101306`) | `wc -l`; `git show 0101306:prng_registry.py \| wc -l` |
| D3 | §1.2 v2.3 "Changed 71 instances across all hybrid kernels" | 10 `[512]` arrays in 9 reverse-hybrid sources survived until `7313a43` (2026-05-03, `[S170-K512]`). The same commit changed the docstring stride line to "(was pos \* k + i)" (`:32`) | `git show 7313a43 -- prng_registry.py`; markers `:1881`, `:1898`, `:2474`, `:2780`, `:2923`, `:3063`, `:3196`, `:3330`, `:3489`, `:3678`; gitignored `prng_registry.py.s170_k512_bak` exists |
| D4 | §3.1 xorshift64 "with Weyl sequence" | xorshift64\*, with no Weyl term | `prng_registry.py:217-232`, `:679-700` |
| D5 | §2.1 every entry carries the template keys | over 44 entries the counts missing are `cpu_reference` 18, `default_params` 18, `seed_type` 12, `state_size` 19. Consumers default `seed_type` to `uint32` (`sieve_gpu_worker.py:180`), which equals the base type in every case | live import; e.g. `:3850-3854`, `:3908-3913` |
| D6 | §6.2 hybrid memory = standard + `skip_sequences` + `strategy_ids` | no `best_skips`; adds `strategy_max_misses`, `strategy_tolerances` and `n_strategies` | `miner/range_miner_worker.py:276-292`; `sieve_gpu_worker.py:248-252` |
| D7 | §8.2 typical strategies 3/5, 5/10, 8/15 | five presets 3/5, 10/20, 5/5, 7/10, 20/50; fallback 3/5. 5/10 and 8/15 occur in no tracked `*.py` | `hybrid_strategy.py:33-79`; `miner/range_miner_worker.py:1098-1109` |
| D8 | §10.1 PyTorch path for "ML scoring in Step 2.5" | executable Step 2 dropped `SurvivorScorer` at v4.0. The live consumer is executable Step 3 | `scorer_trial_worker.py:8`, `:103`; `full_scoring_worker.py:375`; `survivor_scorer.py:570-572` |
| D9 | §10.5 helper sketches | live helpers check torch first; `has_pytorch_gpu` returns False on an unknown family | `prng_registry.py:4141-4180` |
| D10 | §11.1 six-step procedure | omits the Phase-0 encoding pin/version decision and RANGE-MINER builder coverage (RANGE-MINER changed this) | `tests/test_prng_encoding.py:236-243`; `utils/prng_encoding.py:21-24`, `:34`; `miner/range_miner_worker.py:106`, `:142`, `:159`, `:171`, `:189-200`, `:464`, `:474-485` |
| D11 | §14 "Chapter 3: PRNG Registry" | this is Chapter 8 | file name; chapter list in the brief §4 |

---

## 7. RANGE-MINER, the 22-array NPZ contract and the window-anchor change — what they alter in this chapter

- **RANGE-MINER** turns the registry from "the set that runs" into "the set that may run". Coverage is an explicit
  per-variant allow-list validated against the registry (`miner/range_miner_worker.py:105-124`, `:548`, `:628-640`).
  Chapter §16 added.
- **22-array NPZ contract.** The registry's key set fixes the NPZ `prng_type` uint8 id through `utils/prng_encoding.py`
  (Chapter §15 added). Renaming, adding or removing an entry is an interface change.
- **Window-anchor (Brief I).** No kernel changed. The trailing `int offset` now carries `generator_phase`, in the same
  position and dtype (`_generator_phase_tail`, `miner/range_miner_worker.py:295`), pinned to 0 in v1. `window_anchor`
  reaches no kernel argument (Chapter 2 §7.2.1). The four covered forward hybrids with no such argument are
  `PHASE_INCAPABLE_VARIANTS` (`:159-161`). A note was added at chapter §5.1.

---

## 8. Surfaces searched, and surfaces unavailable

- **Governance trail (searched first):** all 140 files `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`, searched with
  `/bin/grep -l "prng_registry\|KERNEL_REGISTRY\|46 PRNG\|44 PRNG\|44-entry\|uncovered famil\|UNCOVERED"` (16 hits). Read:
  `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (`:18-48`, `:86`, `:339`, `:370-387`, `:499`, `:586-588`, `:732-733`).
  `/bin/grep -rn "Option C"` was run for the encoding's docstring claim "TB-approved Option C". **No dedicated `TB_RULING_*`
  file for Option C was found**; this is recorded in §11 and does not affect the count.
- **Planning docs:** `docs/BACKLOG.md` §8.1 (`:155-172`); `docs/TODO_CONSOLIDATED.md` (`:89`, `:220`, `:434`);
  `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1 and D18 (`:1194`); `docs/TFM_PROJECT_FACTS_SKILL.md` §2.6–§2.7 (`:381-411`);
  `docs/PROJECT_FILE_CATALOG.md` (`:287`).
- **Chapters / audits:** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §0, §2–§8, §12–§14 (read, not edited); audits of Chapters 1, 3, 4, 5, 6, 9,
  12, 13, 14 searched (`/bin/grep -n "prng_registry\|_cpu_batch_generate\|prng_encoding\|NotImplementedError\|uncovered"`). Read in
  part: `docs/CHAPTER_4_AUDIT.md` §1, §4 (B2–B7), §5 (S1–S2), §10–§11; `docs/CHAPTER_13_AUDIT.md` §1–§3 (layout).
- **Code:** listed in "files opened" below.
- **Gitignored:** `git ls-files -o -i --exclude-standard`. Relevant hits: `prng_registry.py.s170_k512_bak` (corroborates D3),
  `agent_manifests/definitions.json` (Q6), `__pycache__/prng_registry.cpython-310.pyc`.
- **Git history:** `git log -S 'pos * 512'`, `git log -S 'best_skip_seq[512]'`, `git show 7313a43 -- prng_registry.py`,
  `git show 0101306:prng_registry.py`, `git log --oneline -- prng_registry.py`.
- **Host state:** read-only `ls -la`, `cat` of `agent_manifests/definitions.json`.
- **UNAVAILABLE:** the rigs' deployed copies of `prng_registry.py` (no rig access in this pass); runtime behaviour of any kernel or
  PyTorch path (no runs). Neither is reported as clean.

**Files opened (execution proof):** `docs/CHAPTER_8_PRNG_REGISTRY.md`; `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`
(full); batch-3 common brief; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`; `docs/CHAPTER_13_AUDIT.md`; `docs/CHAPTER_4_AUDIT.md`;
`docs/PIPELINE_BEHAVIOUR_MODEL.md` (§0–§1.1); `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (lines above); `docs/BACKLOG.md` §8.1;
`docs/TFM_PROJECT_FACTS_SKILL.md` §2.6–§2.7; `docs/TODO_CONSOLIDATED.md` (grep); `docs/PROJECT_FILE_CATALOG.md` (grep);
`prng_registry.py` (`:1-152`, `:153-407`, `:408-465`, `:679-700`, `:958-1085`, `:1218-1230`, `:3115-3125`, `:3700-3730`,
`:3767-3790`, `:3830-3875`, `:3905-3920`, `:3957-3990`, `:4130-4235`, plus a signature parse of all 44 kernels);
`utils/prng_encoding.py` (`:1-120`); `sieve_gpu_worker.py` (`:178-310`); `sieve_filter.py` (grep `:46`, `:206`, `:230-233`, `:380-396`);
`miner/range_miner_worker.py` (`:95-300`, `:464-490`, `:628-642`, `:1098-1112`, `:1195-1205`); `hybrid_strategy.py` (`:19-92`);
`survivor_scorer.py` (`:44-48`, `:118-128`, `:568-582`, `:798-810`); `full_scoring_worker.py` (grep `:252-291`, `:375`);
`scorer_trial_worker.py` (grep `:8`, `:103`); `tests/test_prng_encoding.py` (`:1-20`, `:98-102`, `:236-250`);
`agent_manifests/definitions.json` (`:30-50`, `:130-145`, `:205-220`); `tests/test_chapter2_content_gate.py` (run only).

---

## 9. Accurate claims and unverified claims (for the count)

Accurate (35):

1. Purpose: the kernel library for the system. `KERNEL_REGISTRY` is imported by `sieve_filter.py:46`, `sieve_gpu_worker.py`, `miner/range_miner_worker.py:634`, `survivor_scorer.py:45-48` and `utils/prng_encoding.py:30`.
2. Version 2.4 (`prng_registry.py:9`).
3. GPU kernels for forward and reverse sieves as CuPy-compiled CUDA/HIP source: 44 `kernel_source` strings.
4. CPU reference implementations exist (13 functions, `:153-404`).
5. v2.4 PyTorch Phase 1 = `java_lcg`, `java_lcg_hybrid` (`:9-14`, `:95-150`; live `list_pytorch_gpu_prngs()`).
6. §1.2 current state: 2048 local arrays and `pos * k` stride (`:1023`, `:1028`, `:1077`).
7. §1.3 max window 2048 for hybrids (`draw_idx < 2048`, `:1030`).
8. §1.3 seed types uint32/uint64.
9. §2.1 key schema = the exact union of live keys (10 keys).
10. §2.2 naming convention (all 44 names fit the four patterns).
11. §3.1 seed type / state size / description for 10 of 11 rows (xorshift64 → D4).
12. §3.2 the 11 hybrid names and their bases.
13. §4.2 CPU `skip` = outputs skipped before the first (`:176-177`). This is a correct description of the CPU code; the kernel divergence is absorbed (X5).
14. §4.3 `java_lcg_cpu` (`:170-183`).
15. §4.4 `mt19937_cpu_simple` (`:371-404`).
16. §5.1 constant-kernel loop structure (offset pre-advance, burn before first, burn between draws, strict `>` best, `>= threshold`, `atomicAdd`): all 22 constant kernels by signature parse; body at `:958-1004`.
17. §6.1 standard buffer table (`sieve_gpu_worker.py:198-202`).
18. §6.2 `skip_sequences` n_seeds×k uint32, `strategy_ids` n_seeds uint32 (`sieve_gpu_worker.py:251-252`).
19. §6.3 per-thread 2048 arrays (`:1023`, `:1028`).
20. §6.4 stride `pos * k` (`:1077`); a 512 stride existed earlier (`git log -S 'pos * 512' -- prng_registry.py` → `0101306`, `7313a43`).
21. §7.1 xorshift32 kernel (`:408-464`).
22. §7.2 java_lcg kernel (`:958-1004`).
23. §8.1 hybrid body: `expected_skip = 5`, window, re-centre, abort, best strategy, guard (`:1005-1084`).
24. §8.2 meaning of `max_consecutive_misses` / `skip_tolerance` (`hybrid_strategy.py:19-20`).
25. §9.2 host-side reversal in `sieve_filter.py` (`:231-233`, `:394-396`).
26. §9.3 reverse signature `candidate_seeds … int offset`, no `a`/`c` (`:3115-3126`).
27. §10.2 lazy torch import (`:78-89`).
28. §10.3/§10.4 `java_lcg_pytorch_gpu` (`:95-138`).
29. §11.2–§11.4 templates match the shape of existing CPU references, constant kernels and registry entries.
30. §12.1 `get_kernel_info` (`:4187-4191`).
31. §12.2 `list_available_prngs` (`:4192-4194`).
32. §12.3 `get_cpu_reference` body (`:4195-4197`).
33. §13.1 the 44 rows: names, direction, variable-skip column.
34. §13.2 default parameters (live `default_params` for xorshift32, pcg32, lcg32, java_lcg, minstd; `{}` for mt19937, xoshiro256pp, philox4x32, sfc64).
35. §14 key features: CuPy kernels, CPU references, PyTorch, 2048 windows, multi-strategy hybrid.

Unverified (2), each with its final placement:

- **U1** — §1.2 "Eliminated illegal memory access causing GPU crashes". This is a runtime claim; this pass did no runs. **Final placement: UNVERIFIED** (stays open).
- **U2** — §1.2/§10.1 PyTorch "works on both CUDA and ROCm". The code is device-string agnostic on reading, but it was not run on ROCm. **Final placement: UNVERIFIED** (stays open).

Absorbed (17; not counted separately):

- X1 §5.2 "All kernels use a 3-lane check" → Chapter 2 §6.2.1 (43 of 44; the `mt19937_hybrid` single-lane exception).
- X2 §5.2 "Why 3 lanes?" (%125 "covers different bit patterns") → Chapter 2 §6.3–§6.4, F-1 (CRT-redundant).
- X3 §5.1 "`offset` (always LAST)" and the §8.1 template's `int offset` → Chapter 2 §7.3 / skill §2.7 #5.
- X4 §8.1 "test each strategy, keep best" as a description of every hybrid → Chapter 2 §5.3 (reverse hybrid returns on the first strategy that clears; e.g. `:3720`).
- X5 §4.1 "Verification — validate GPU results" → Chapter 4 S1 / `docs/BACKLOG.md` §8.1 / Chapter 2 §12.1.
- X6 §4.1 "Fallback — when GPU unavailable" → Chapter 4 B4.
- X7 §3.3/§9.1 "backward validation of candidates from forward sieves" → Chapter 2 §4.2 (independent passes, then intersection) and §3.2 (reversed target, forward generator); `miner/range_miner_worker.py:1203-1205`.
- into **D1** (6): §1.3 table totals; §3.1 "(14 total)"; §3.2 "(14 total)"; §3.3 "(18 total)" and its table missing xoshiro256pp/sfc64; §13.1 "All 46"; §14 summary-table counts.
- into **D5** (2): §4.2 "common signature" for every CPU function (`mt19937_cpu` has no `**kwargs`, `:335`); §12.3 implied to work for every family (`KeyError` on 18).
- into **D11** (2): the "Next Chapter: Chapter 4: Meta-Optimizer (Step 2.5)" pointer; the "End of Chapter 3" line.

---

## 10. Count reconciliation

| Bucket | Count | Items |
|---|---|---|
| Accurate | 35 | §9 items 1–35 |
| BROKEN | 0 | — |
| SWITCHED-OFF-BY-RULING | 1 | S1 |
| DESCRIPTION-ONLY | 11 | D1–D11 |
| UNVERIFIED | 2 | U1, U2 |
| Absorbed into an existing item, not counted separately | 17 | X1–X7 (7, into Chapter 2 / Chapter 4 items); into D1 (6); into D5 (2); into D11 (2) |
| **Total claims checked** | **66** | 35 + 0 + 1 + 11 + 2 + 17 = 66 |

No UNVERIFIED item was resolved in this pass. U1 and U2 keep the explicit placement UNVERIFIED.

---

## 10a. Chapter edit record and clean control

- Chapter `docs/CHAPTER_8_PRNG_REGISTRY.md`: 1,058 → 1,525 lines (`wc -l`). All changes were made with the Edit tool. Every
  pre-existing heading is still present (heading diff against `git show a15b524:…`: no heading removed).
  `git diff --numstat` shows 4 lines removed. Three are header/footer lines replaced by annotated versions that keep the
  original wording (Version, Lines, "End of Chapter 3"). The fourth is a `---` separator that git re-aligned: there are 17
  `---` separators before and after.
- This audit file was created once with Write. Every later change used Edit.
- Clean control: `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` unmodified (`git diff --quiet`). `python3 tests/test_chapter2_content_gate.py`
  → "12/12 checks green (6 gates + 6 mutants) · SENTINEL : PASS".
- Completion sentinel for Chapter 8: **PASS**.

## 11. Observations outside the claim count (not chapter claims; recorded, not labelled)

- `utils/prng_encoding.py:13` calls the alphabetical scheme "TB-approved Option C, 2026-05-xx". The RANGE-MINER spec
  §5.4 item 2 (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:386`) says "Legacy `0=java_lcg`, `1=java_lcg_reverse` are
  preserved". Live alphabetical ids give `java_lcg` = 0 and `java_lcg_hybrid` = 1. No `TB_RULING_*` file recording
  Option C was found. Whether `:386` was superseded by Option C is for the owner. It is not a Chapter 8 claim.
- The spec (`:387`) describes `randu`/`randu_reverse` as decode-only aliases. The live module has no alias table: it is
  purely registry-derived (`utils/prng_encoding.py:42-44`), and `randu` is not a registry key. Not a Chapter 8 claim.
- The `prng_registry.py` docstring repeats its "Version 2.4" block twice (`:9-14`, `:16-21`). The `__main__` banner prints "v2.1"
  and "Hybrid (NEW!) … [+3 …]" (`:4199`, `:4230`). Cosmetic.
- `java_lcg` and `java_lcg_hybrid` each write the `'pytorch_gpu'` key twice in the dict literal (`:3961-3962`, `:3974-3975`). Harmless.
- PWC's fallback literals for lcg32 (`a=1664525, c=1013904223, m=0xFFFFFFFF`, `sieve_gpu_worker.py:224-226`) differ from
  the registry's `default_params` (`1103515245, 12345, 0x7FFFFFFF`). They are used only when a key is missing, and the
  registry supplies all three, so they are inert. PWC is non-certifying.
- The proposal (`:370`) records PWC's generic fallthrough (`sieve_gpu_worker.py:304`) as a pre-existing bug for uncovered
  families. It is already governed, and RANGE-MINER's fail-fast is its resolution on the certifying path.

## 12. Updates needed elsewhere (listed, not made)

- `docs/PROJECT_FILE_CATALOG.md:287` (Chapter 8 row): "unaudited / claims 4,323 lines" → audited 2026-11-11; registry is 4,235 lines, 44 entries.
- `docs/PIPELINE_BEHAVIOUR_MODEL.md` D18 (`:1194`): the Chapter 8 half is now corrected in the chapter.
- `docs/TFM_PROJECT_FACTS_SKILL.md`: nothing contradicts it; optionally record that PWC/miner coverage (`CLAUDE.md` §7) was re-confirmed at `a15b524`.
- `CLAUDE.md` §7: confirmed accurate. The anchor `sieve_gpu_worker.py:208-306` is valid; the branches sit at `:217-304`.
- `docs/BACKLOG.md` §1: no new repair item from this chapter (0 BROKEN). S1 goes to the owner decision list.
