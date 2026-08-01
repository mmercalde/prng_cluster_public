# TEAM ALPHA → TEAM BETA — Chapter 2 is recoverable; repair type should be re-scoped

**Re:** `docs/CHAPTER_2_SOURCE_MAP_v1.md`. Reconnaissance only — no code, config or
documentation changed; nothing executed on a GPU; no Chapter 2 text written.

**Beta ruled:** `Chapter 2 status: MISSING CORE CONTENT / repair type: reconstruction / not:
stale-text correction.`

**That ruling was made on incomplete information.** The chapter is not missing. It exists in
git history at 743 lines and was destroyed by a stale-copy overwrite. Alpha requests the
repair type be re-scoped from **reconstruction** to **restore-and-audit**.

---

## 1. The forensics

```
53a3829  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   709 lines  (§1-13 + Summary)
d14dcdd  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   743 lines  (+ §14)
         CHAPTER_2_BIDIRECTIONAL_SIEVE.md         34 lines  (root, §14-only fragment)

248e48c  "chore: move CHAPTER docs to docs/ folder"
           D  CHAPTER_2_BIDIRECTIONAL_SIEVE.md         (-34)
           M  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md    (-709)   ← the chapter
         → docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   34 lines
```

Verified by `git log --follow`, `git show <sha>:docs/…`, and `git show 248e48c --name-status`.
**A housekeeping "move" commit copied the 34-line root fragment over the 743-line chapter and
deleted the root file.** No error surfaced. The full text is retrievable today:
`git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`.

### 1.1 This is the second recorded instance of one defect class

| # | commit | what it overwrote | how it was found |
|---|---|---|---|
| 1 | `2389b61` (07-07) | the Optuna threshold fix from `3fdf434` — rewrote the file from a **pre-fix copy** while doing unrelated PRNG-encoding work; commit message never mentions thresholds | targeted audit, four months later |
| 2 | `248e48c` | the 743-line Chapter 2 — copied a same-named 34-line fragment over it during a folder move | this reconnaissance |

Both were **silent**. Both were found only because someone went looking for something else.
Alpha submits this is a **pattern**, not two incidents, and that it warrants naming as such:
*a file operation that replaces content with a same-named stale copy produces no diagnostic
and no gate fires.* Skill §2.7 #2 records the first; the second is now evidence it recurs.

## 2. What re-scoping changes

**Cost and risk both fall.** Reconstruction implies authorship from live code against a blank
page. Restore-and-audit is: recover the text, verify each section against current source,
re-scope what is engine-obsolete.

**Recovered section structure** (`d14dcdd`):

| § | title | disposition Alpha proposes |
|---|---|---|
| 1 | Mathematical Foundation | likely substantively correct — verify against the whitepaper |
| 2 | Forward Sieve | verify; engine references need re-scoping |
| 3 | Reverse Sieve (incl. **3.2 "Same PRNG, Different Direction"**) | likely correct — this is skill §0.2's content |
| 4 | Bidirectional Intersection | likely correct |
| 5 | Skip/Gap Handling | **restore and extend — see §3 (G-1)** |
| 6 | **Three-Lane CRT Architecture** | **restore — see §3 (G-5)** |
| 7-13 | Architecture · ROCm setup · `GPUSieve` · `run_sieve` · `run_hybrid_sieve` · CLI · Integration | **re-scope, do not restore** — all pre-S172, describing `sieve_filter.py`/`GPUSieve` as the engine. RANGE-MINER did not exist |
| 14 | Inter-Chunk GPU Cleanup | retain, corrected |

Every recovered line is **pre-S172**. §7-13 describe a superseded engine and must be rewritten
against RANGE-MINER; §1/§3.2/§4/§6 are the sections whose loss actually cost the project
knowledge.

## 3. Two gaps that outrank the chapter itself

**G-5 — the three-lane CRT test is live in every kernel and explained nowhere current.**
`(output % 1000) && (output % 8) && (output % 125)` appears at `prng_registry.py:984-986`,
`:1042-1044`, `:3146-3148`. **The only prose explanation in the project's history is §6 of the
deleted chapter.** It has been undocumented since `248e48c`.

**G-1 — "why skip exists" has no repository document at all.**
The physical model (two unpublished pre-test draws before every live draw; per-session
auditor-verified equipment selection; evening co-drawing of D3/D4/Fantasy 5/Daily Derby) exists
in the project **only** as `tfm-project-facts` §0.4 — which states outright that it is "the
part nobody had written down." The primary source, the *California State Lottery Daily &
SuperLotto Plus Draw Procedures* (eff. 2021-06-09), **is not in the repository.**

The audit's own assessment:

> This absence already caused Alpha, Beta and Claude Code to independently recommend deleting
> `skip_min`/`skip_max`. **Chapter 2 §5 is the natural home for it.** Writing it there is
> arguably the highest-value paragraph in the whole reconstruction.

**Alpha requests a ruling on committing the draw-procedures PDF to `docs/`.** At present the
primary evidence for a foundational design decision — one already argued over three times —
exists nowhere under version control. Alpha notes both remotes are effectively public and the
document is a public state record.

**G-2 — the binding session-stream ruling has no ruling document.**
Per-session ordering normative / combined-container order carrying no PRNG-advance meaning /
combined-session sequential sieve non-certifying / reorder migration cancelled — carried
in-repo **only** by the skill's §2.10 summary. Every other adjudicated area has a
`TB_RULING_*` file. `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` is the request,
not the ruling. **Alpha requests Beta issue the ruling document** so Chapter 2 can cite an
authority rather than a summary.

## 4. A trap in Wall C's obvious candidate

Source 9 (Beta's bounded independent known-answer control) is **NOT FOUND** — one-line
specification, no artifact. The obvious reference implementation is a trap:

- `java_lcg_cpu` (`prng_registry.py:170-183`) applies `skip` **once before generating**.
- The kernel applies it **between every draw** (`:987-989`).
- **They agree only at `skip = 0`.**

Anyone building Wall C on `cpu_reference` would validate the wrong semantics — and would do so
in the deliverable specifically intended to catch semantic error. The usable seeds are
`create_java_lcg_test.py` (constant, sequence-correct) and
`create_java_lcg_variable_test.py` (variable); neither is a harness, and neither's output JSON
is in the tree.

## 5. Also established

- **`docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md`** — PARTIALLY USABLE, high value.
  Staleness checked by diff rather than re-reading: since its survey commit `42a7229` the only
  non-test/non-doc sources changed are `miner/*`, `window_optimizer_integration_final.py` and
  `persistent_worker_coordinator.py`; `sieve_filter.py`, `sieve_gpu_worker.py`,
  `prng_registry.py`, `coordinator.py` and `utils/*` are **byte-unchanged**, so its anchors
  hold. Two caveats: its §8.5/O7 ("Optuna thresholds never reach a kernel") is **superseded by
  `8a55a68`**, and it excludes `miner/` by its own instruction — so it covers the
  *non-certifying* half of the subject. **It is untracked and would be lost by a clean
  checkout.**
- **Whitepaper-vs-chapter boundary** — the whitepaper says *why* bidirectional sieving works;
  the chapter says *what the system does when it runs*. One real gap: whitepaper §4 uses
  `G(s,−i)` (a backward step) while the implementation evaluates `G(s,i)` forward against
  `residues[::-1]`. The chapter must state the implemented construction and name the
  divergence without adopting the whitepaper's notation. **Alpha flags this as a mathematical
  question for Beta**, not a documentation choice.
- **G-4** — the miner's kernel-ABI-by-variant contract (forward and reverse constant kernels
  do **not** share arg layout; every fixed-skip reverse kernel hardcodes its generator
  parameters in the body; arity varies 12/14/15/17 by variant and family) is documented only
  in code comments and belongs in the chapter.

## 6. Rulings requested

1. **Re-scope Chapter 2 from *reconstruction* to *restore-and-audit*** — recover `d14dcdd`,
   verify §1-6 and §14 against live source, rewrite §7-13 against RANGE-MINER.
2. **Commit the CA draw-procedures PDF to `docs/`?** (G-1)
3. **Issue the session-stream ruling document** so Chapter 2 cites an authority, not a skill
   summary. (G-2)
4. **Name the stale-copy-overwrite pattern** — two recorded instances, both silent, both found
   by accident. Does it warrant a standing gate or checklist item?
5. **`G(s,−i)` vs `G(s,i)` on reversed residues** — how should the chapter state the
   relationship without contradicting the whitepaper?
6. **Wall C reference semantics** — given the `cpu_reference` skip divergence, what should the
   known-answer control be built from?

## 7. VIR declaration

Sentinel **INCOMPLETE, deliberately**: 7/9 sources COMPLETE, Source 5 PARTIAL (the binding
ruling exists only as a skill summary — G-2), Source 9 **NOT FOUND** (G-3). Fault-injection
control **n/a for read-only reconnaissance, stated rather than omitted.** VIR-6 scope: repo +
VM 101 only. The four files owned by the concurrent Chapter 1 P0 session were **not opened**;
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1315` records Chapter 2's own intended scope and is a
**dependency for the reconstruction pass**. All three CT100s ping UP but **deployed rig kernel
source was not compared to VM 101** — every kernel claim here is about the VM 101 tree only.
