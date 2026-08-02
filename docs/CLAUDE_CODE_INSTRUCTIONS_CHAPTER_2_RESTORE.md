# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_RESTORE.md — REV1

**Chapter 2 (Bidirectional Sieve) — restore and audit.**

**This is recovery, not authorship.** The chapter exists at **743 lines** in git history and was
destroyed by a stale-copy overwrite. Team Beta ruled the repair type **restore-and-audit**, not
reconstruction.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

**Concurrency:** a Resolved Execution Set session may be running. It touches
`miner/`, `agents/watcher_agent.py` and `window_optimizer.py`. **This brief edits
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` and reads everything else.** No collision expected;
report one if you see it.

---

## 0. What happened, and what to recover

```
53a3829  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   709 lines  (§1-13 + Summary)
d14dcdd  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md   743 lines  (+ §14)
248e48c  "chore: move CHAPTER docs to docs/ folder"
           D  CHAPTER_2_BIDIRECTIONAL_SIEVE.md         (-34)
           M  docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md    (-709)   ← the chapter
```

A housekeeping "move" copied the **34-line root fragment over the 743-line chapter** and deleted
the root file. Recover with:

```
git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md
```

**Reconnaissance is already done** — `docs/CHAPTER_2_SOURCE_MAP_v1.md` maps every source. **Cite
it; do not redo it.**

## 1. Section disposition

| § | title | action |
|---|---|---|
| 1 | Mathematical Foundation | **verify** against `BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` |
| 2 | Forward Sieve | **verify**; re-scope engine references |
| 3 | Reverse Sieve (incl. 3.2 "Same PRNG, Different Direction") | **verify** — this is skill §0.2's content |
| 4 | Bidirectional Intersection | **verify** |
| 5 | Skip/Gap Handling | **restore and EXTEND — see §2** |
| 6 | **Three-Lane CRT Architecture** | **restore — see §3** |
| 7-13 | Architecture · ROCm setup · `GPUSieve` · `run_sieve` · `run_hybrid_sieve` · CLI · Integration | **RE-SCOPE, do not restore verbatim** — all pre-S172, describing `sieve_filter.py`/`GPUSieve` as the engine. RANGE-MINER did not exist |
| 14 | Inter-Chunk GPU Cleanup | **retain**, corrected |

**Every recovered line is pre-S172.** §1/§3.2/§4/§6 are the sections whose loss actually cost
the project knowledge; §7–13 describe a superseded engine.

## 2. §5 — the highest-value content in the whole restore

**Two things must be written here that exist nowhere in the repository.**

**(a) Why skip exists — the physical model.** Per the *CA Lottery Daily & SuperLotto Plus Draw
Procedures* (eff. 2021-06-09): **two pre-test draws before every live draw**, generated,
verified, certified and **never published** (§V); **equipment selected per session** by an RNG
program, auditor-verified (§II); the evening session draws **D3, D4, Fantasy 5 and Daily Derby
together**. The observable sequence therefore has **real structural gaps of varying size.**

This is the paragraph whose absence caused Team Alpha, Team Beta and Claude Code to
**independently recommend deleting `skip_min`/`skip_max`** — a cornerstone — because all three
inferred intent from kernel signatures that were themselves the defect. **Write it so no future
reader re-derives "remove it."** (The PDF itself is not in the repo — an open item.)

**(b) Michael's design intent, which is corroborated but undocumented.** The goal was never to
reverse state; it is to extract a **fingerprint**. The published sequence reveals only fragments
of PRNG state before other outputs interleave, so variable skip exists to **find the windows
where coherent skip structure surfaces**, and to produce survivors with *varied* skip structure
so tree/NN models have something to rank on. Corroborated on three of four elements
(`instructions.txt:1247`; the survivor-as-(seed, skip_hypothesis) pair in the **deleted §5.4** at
`d14dcdd`; the Oct-2025 `pattern_stats` spec) — **the framing itself is NOT FOUND anywhere.**
This is its home.

**Also record, from `SKIP_SEMANTICS_SEARCH_v1.md`:** `skip_min`/`skip_max` are documented in
**two readings at two pipeline stages** — element-wise **input** bound (`instructions.txt:1182-1183`)
and **output** ML statistic (`PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158`). **Not a
contradiction.** And the defect callout: hybrid kernels do not execute the requested semantics;
**hybrid optimization results are non-certifying.**

## 3. §6 — the three-lane CRT test

`(output % 1000) && (output % 8) && (output % 125)` is live in **every** kernel
(`prng_registry.py:984-986`, `:1042-1044`, `:3146-3148`). **The only prose explanation in the
project's history is §6 of the deleted chapter.** It has been undocumented since `248e48c`.
Restore it and verify it against the live kernels.

## 4. §7–13 — re-scope against RANGE-MINER

Do **not** restore these verbatim. The engine is now RANGE-MINER: persistent per-GPU daemons,
coordinator/worker stripe lifecycle, D5/D6 assembly, the frozen 22-array NPZ contract, the
finalizer. **Cite rather than duplicate:** `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`,
the D5/D6 changelogs, `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`, `HYBRID_SKIP_BOUND_AUDIT.md`,
`DAILY3_CONSUMER_CONTRACT_v1.md`, and the **bounded Phase 6 submission** (`d98298c`).

**State the current certified position:** bounded Phase 6 is **certified and closed** — Wall A's
full consumer chain, Wall B's node-assignment independence across two ROCm rig pairs, and the
Miner Known-Answer Transfer Gate at 8/8 populations exact-set equal. **With its scope limit:**
Wall A/B used **constant-skip** generations; **hybrid worker semantics are covered by the
transfer gate, not by a four-phase Wall-A consumer run.**

## 5. Boundaries

**Whitepaper vs chapter:** the whitepaper says *why* bidirectional sieving works (survival
probability, the squared exponent, why loose thresholds are required); the chapter says *what
the system does when it runs.* **One real divergence to state without adopting the whitepaper's
notation:** whitepaper §4 uses `G(s,−i)` — a backward step — while the implementation evaluates
`G(s,i)` **forward** against `residues[::-1]`. Describe the **implemented** construction and
name the divergence.

**`docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md`** is PARTIALLY USABLE and now tracked.
Two caveats: its §8.5/O7 ("Optuna thresholds never reach a kernel") is **superseded by
`8a55a68`**, and it **excludes `miner/`** by its own instruction — so it covers the
*non-certifying* half. Use it, cite it, do not inherit its stale claim.

## 6. Out of scope

- **Do not modify any code**, test, config or manifest. This is a documentation deliverable.
- Do not fix the hybrid skip wire-in, the `java_lcg_cpu` non-zero-skip mismatch, or any §2.7
  dead dimension. **Describe them; do not repair them.**
- Do not touch other chapters.
- Do not re-derive the source map, the threshold audit, the skip-semantics search or the Phase 6
  evidence — **cite them.**
- Do not propose removing anything (skill §0.4).

## 7. Verification-integrity controls (VIR-1…6)

- **execution proof** — every retained claim carries a `file:line` verified this session or a
  named audit citation. **A recovered line is not a verified line.**
- **clean control (VIR-2)** — state which recovered sections you verified **correct and
  unchanged**. A report listing only corrections gives no evidence the rest was checked.
- **fault-injection control** — n/a for a documentation pass; **say so** rather than omitting it.
- **completion sentinel (VIR-3)** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` plus a
  per-section coverage table.
- **unavailable-observer (VIR-5)** — anything unverifiable is `UNAVAILABLE`, not assumed correct.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces.

## 8. Report

The recovery command and what it produced. Per-section: verified / corrected / re-scoped, with
anchors. The §5 additions and §6 restoration called out specifically — they are the reason this
deliverable exists. What was cited rather than duplicated. The completion sentinel. Then STOP.
**Do not commit.**
