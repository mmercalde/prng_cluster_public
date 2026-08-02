# SESSION CHANGELOG — 2026-08-02 (Chapter 1 & 2 closure)

**Focus:** Verify Chapters 1 and 2 against HEAD and add a closure statement to each, per
`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_AND_2_CLOSURE.md` (REV1). Documentation only.
**Outcome:** Both chapters closed. `CHAPTER 1 CLOSURE: PASS`, `CHAPTER 2 CLOSURE: PASS`.
Executable gate green 12/12. Two documentation files changed; no code, tests, config or manifests.

---

## Summary

Chapter 1's anchors had drifted again since its `40c3c83` correction pass — bounded Phase 6
(`d98298c`) added +173 lines to `window_optimizer_bayesian.py` and the Resolved Execution Set
(`63e627f`) plus admission binding (`eff6616`) added +161 to `window_optimizer.py`. ~26 anchor
groups were corrected, the §4.1 bounds snapshot was machine-regenerated (bounds **unchanged** —
`configuration_digest` byte-identical), and two new sections were written: **§8.1.2** (the
sampler-neutral core) and **§3.1.2** (absorbing Chapter 2's F-4 into audit conflict C-2).

Chapter 2's one open item — §6.2's unreproducible "39 occurrences of the lane test" — is
**settled at 43**, with the counting method published as executable code.

Two surprises. First, the mandated gate initially returned **FAIL 8/12** for a purely
environmental reason (the whole rig fleet was powered off, and the fail-closed P0.5 dataset
preflight correctly refused); this was proven environmental rather than asserted, and the
underlying fact — *four of the twelve gate arms require a reachable fleet* — is now recorded in
the chapter. Second, one fact the brief asked me to **confirm** (GridSampler unconstructibility)
turned out to be **absent from the chapter entirely**, so it was derived live and added.

---

## Work Completed

| Item | Status | Notes |
|------|--------|-------|
| Ch1 — re-verify every `file:line` anchor against HEAD `81ef3f1` | ✅ | ~26 anchor groups corrected. Drift was systematic: `window_optimizer.py` grew **only inside `main()`**, so everything above `:1239` was unaffected. Symbol-first citation convention applied where a line is unstable |
| Ch1 — regenerate §4.1 bounds snapshot | ✅ | By `scripts/extract_search_bounds_snapshot.py`, **not hand-edited**. `repository_commit` `0c47fe3…` → `81ef3f1…`; `configuration_digest` **byte-identical** `sha256:6077bb1a…2747cc` — the bounds did not move |
| Ch1 — document the sampler-neutral core | ✅ | New **§8.1.2**. `run_optimization(..., *, sampler, sampler_metadata)` — both **required and keyword-only, no default**. `SAMPLER_ENTRYPOINTS` **deliberately unwired**; autonomous sampler selection is reserved authority |
| Ch1 — confirm gated strategies still described as gated, not deleted | ✅ + extended | §6.4/§10.1 verified correct. **The `GridSampler` unconstructibility fact was missing and was added** (see Decisions) |
| Ch1 — absorb Chapter 2's F-4 into C-2 | ✅ | New **§3.1.2**. Settles C-2 as an **observed inconsistency, NOT the repair** |
| Ch2 — settle the §6.2 lane-test count | ✅ | **43**, in 43 of 44 kernels. New **§6.2.1** with the method as runnable code |
| Ch2 — §1.1 factual corrections | ✅ | `:1004` → `:3963-3966`; "three non-parameterised kernels" → **two** (`:3125-3126`, `:3182-3183`), both *reverse* kernels |
| Ch2 — discharge VIR-6 unavailable-surface #1 | ✅ | The concurrent session landed; §5.4/§7.2/§8.4 anchors re-read at `81ef3f1` — **none moved** |
| Ch2 — amend VIR-6 unavailable-surface #4 | ✅ | Superseded by §6.2.1; the residual unavailability (40 kernel **bodies** unread) retained explicitly |
| Closure statements, both chapters | ✅ | Ch1 **§17**, Ch2 **§14**: verified-against / what is verified / what remains open and where tracked / what the chapter is NOT / sentinel |
| Executable gate (`tests/test_chapter1_p0_corrections.py`) | ✅ | **12/12, `SENTINEL : PASS`**, including mutants M5 and M6 |

---

## Files Created/Modified

| File | Action | Destination |
|------|--------|-------------|
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` | Modified | VM 101 `~/distributed_prng_analysis/docs/` — 1857 → **2303** lines; new §17, new §8.1.2, new §3.1.2, TOC completed (15–17), header status |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | Modified | VM 101 `~/distributed_prng_analysis/docs/` — 1208 → **1441** lines; new §14, new §6.2.1, §1.1 corrections, VIR-6 surfaces 1 and 4, version → 4.2.0 |
| `docs/SESSION_CHANGELOG_20260802_CHAPTER_1_AND_2_CLOSURE.md` | Created | VM 101 `~/distributed_prng_analysis/docs/` |

**Diffstat:** `2 files changed, 822 insertions(+), 143 deletions(-)` (before this changelog).

**Out of scope and untouched, as briefed:** no code, tests, config or manifests — not
`window_optimizer.py`, not `window_optimizer_bayesian.py`, not `execution_set.py`. No F-item,
dead dimension or C-item repaired. Nothing removed or proposed for removal.

---

## Issues / Incidents

| Issue | Resolution |
|-------|------------|
| **Gate returned `SENTINEL : FAIL` 8/12.** `G-FLAG-FAILCLOSED`, `G-STRATEGY-FAILCLOSED`, `M1`, `M2` all failed on one assertion — *"clean control: bayesian did not reach `run_bayesian_optimization`"* — because the P0.5 dataset preflight refused with `No route to host` to `.122`/`.156`/`.164` | **Environmental, not a regression — and proven, not asserted.** Ping sweep showed **all six** rig addresses down (Proxmox CT set *and* bare-metal set): the fleet was powered off, a state distinct from the "booted into Proxmox" condition CLAUDE.md §3 records. Once the fleet was restored the **same edited chapter, same gate, same commit** went **8/12 → 12/12**. See Decisions for the three-way proof |
| **`pytest` is not installed** in `~/venvs/torch` | The gate is a standalone harness; run as `python3 tests/test_chapter1_p0_corrections.py`. Not a defect — recording it so the next session does not chase it |
| One Chapter 2 anchor (`known_answer_reference.py:409-411`) was path-abbreviated | Fully qualified to `tests/phase6/known_answer_reference.py:409-411`. Content was correct; the path was inherited from the preceding citation in the same table cell |

---

## Decisions Made

- **The §6.2 count is 43, and 39 is withdrawn.** The three candidate counts measure genuinely
  different things: **43** = complete three-lane conjunctions (the test §6.2 actually prints);
  **31** = the subset carrying the `(unsigned int)` cast — a formatting variant, semantically
  inert under C integer promotion; **30 + 13** = the same 43 partitioned by loop-index name.
  **39 is reproducible by no method.** The chapter now publishes the counting program itself,
  because the withdrawn 39 survived precisely by being carried as a bare figure with no stated
  way to re-derive it. *A number in a chapter must be reproducible by the method the chapter
  names.*

- **"One per kernel" retired.** It is 43 of **44** kernels. `mt19937_hybrid_multi_strategy_sieve`
  (`prng_registry.py:773`) tests mod 1000 only (`:820-821`). **This is not a defect** — §6.3
  already proves the three-lane test is *exactly equivalent* to that single comparison, so the
  mt19937 kernel is the same predicate without the two redundant conjuncts. It is also outside
  TFM's sieve path (java_lcg only).

- **The gate's fleet dependency is recorded as a standing fact, not a one-off incident.** Proven
  three ways: (1) *empirically, single-variable* — fleet up, same edited chapter → 12/12;
  (2) *clean control* — the **pristine** chapter at `81ef3f1` via `git stash`, fleet up → also
  12/12, so the edits neither introduce nor mask a failure; (3) *structurally* — only
  `gate_snapshot_extracted`, `gate_skip_defect_note`, M5 and M6 ever open the chapter path
  (`tests/test_chapter1_p0_corrections.py:64`, `:578-579`, `:646`, `:680-681`, `:707`); the four
  arms that were red **never read the chapter file at all**. **Consequence now in Ch1 §17.2: a
  green 12/12 certifies the chapter *and* asserts a reachable fleet; a red one does not by itself
  indict the chapter.** Observation for the gate owner — **not** a proposal to change the gate.

- **GridSampler unconstructibility was added, not merely confirmed.** The brief listed it as a
  fact to check; it was absent from the chapter. Derived live rather than transcribed:
  `GridSampler.__init__` eagerly materialises `list(itertools.product(...))` and then *shuffles
  it* (verified by `inspect.getsource` against installed **optuna 4.4.0**), so the cost is paid at
  construction. At live §4.1 bounds: `45 × 101 × 3 × 11 × 241 × 46 × 46 = 76,485,750,660`
  ≈ **7.649 × 10¹⁰** points; at a **measured** 104 bytes/point ≈ **7.23 TiB** resident.
  Framed as: the documented four-sampler design is **not implementable as stated for Grid**
  without an explicitly coarsened grid — a **design decision with governance consequences, for
  Beta**, not a restoration. Explicitly **not** an argument for deletion (§0.4).

- **Three manifest findings recorded as open, not repaired** (Ch1 §10.1, carried into §17.3).
  The load-bearing one: **`parameter_bounds` in the manifest is a live admission gate looser than
  §4.1's authority** — `agents/step_runner/command_builder.py:151-170` validates against it, and
  it permits `window_size = 2` where the S172 TB ruling raised the search floor to **6**. It
  cannot widen the Optuna search space, but it **would accept a WATCHER-proposed `window_size=2`**
  — precisely the value S172 excluded as chance-driven. Traced producer → consumer before being
  asserted.

- **"Closed" means verified-and-bounded, not finished.** Both closure statements enumerate every
  open item with where it is tracked — Ch1 §17.3 carries 14, Ch2 §14.3 carries F-3/F-4/F-5/F-7,
  the six §12.1 inherited items and four still-`UNAVAILABLE` VIR-6 surfaces. *A closure statement
  that hides an open item is worse than no closure statement.*

- **Chapter 2's fault-injection control is `NOT_APPLICABLE`, and the reason is itself an open
  item.** **No executable gate covers Chapter 2** — `tests/` has `test_chapter1_p0_corrections.py`
  and no Chapter 2 equivalent, verified this session. Chapter 1's edits were gated and run;
  Chapter 2's could not be. Recorded in Ch2 §14.3 as a governance observation for the gate owner.

---

## Verification performed

```bash
cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate

# Executable gate — run three times: edited/fleet-down, edited/fleet-up, pristine/fleet-up
python3 tests/test_chapter1_p0_corrections.py          # final: 12/12, SENTINEL : PASS

# Clean control (edits stashed, then restored; hash verified identical after pop)
git stash push -m ch1-closure-edits-temp docs/CHAPTER_1_WINDOW_OPTIMIZER.md
python3 tests/test_chapter1_p0_corrections.py          # pristine: also 12/12
git stash pop

# §6.2 count, the method the chapter now names
python3 -c "
lines=open('prng_registry.py').read().split(chr(10))
print(len([i for i,l in enumerate(lines) if '% 1000' in l
      and '% 8' in chr(10).join(lines[i:i+3]) and '% 125' in chr(10).join(lines[i:i+3])]))"   # 43

# Anchor range sweep, both chapters — 146 explicit file:line anchors, 0 out of range, 0 missing
# Concurrent-session interval check
git diff --stat eed3904..81ef3f1 -- miner/ agents/ window_optimizer.py prng_registry.py
```

Fleet state at the final gate run: `.122` / `.156` / `.164` **UP** (Proxmox CT set), key auth
verified to all three, frozen dataset present. `.120` / `.154` / `.162` down — expected, since a
machine runs one OS at a time (CLAUDE.md §3).

---

## Git Commands

> **Not run by the agent.** Per CLAUDE.md §1.1/§1.2 Michael commits and dual-pushes.

```bash
cd ~/distributed_prng_analysis
git add docs/CHAPTER_1_WINDOW_OPTIMIZER.md \
        docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md \
        docs/SESSION_CHANGELOG_20260802_CHAPTER_1_AND_2_CLOSURE.md

git commit -m "docs: close Chapter 1 and Chapter 2 — verified against 81ef3f1

Chapter 1 (§17 closure, sentinel PASS): ~26 anchor groups re-verified and
corrected after Phase 6 (+173 bayesian) and the execution set / admission
binding (+161 window_optimizer, all inside main()); §4.1 snapshot machine-
regenerated (bounds unchanged, digest byte-identical); new §8.1.2 sampler-
neutral core (sampler/sampler_metadata required and keyword-only;
SAMPLER_ENTRYPOINTS deliberately unwired); new §3.1.2 absorbing Chapter 2's
F-4 into audit conflict C-2 as an observed inconsistency, NOT the repair;
GridSampler unconstructibility added (7.649e10 points, ~7.23 TiB at
construction, derived live against optuna 4.4.0).

Chapter 2 (§14 closure, sentinel PASS): §6.2's '39 occurrences' withdrawn as
unreproducible and settled at 43, in 43 of 44 kernels, with the counting
method published as runnable code (§6.2.1); mt19937_hybrid_multi_strategy_sieve
named as the single-lane exception; §1.1 default-params anchor corrected to
:3963-3966 and the a/c hardcode count corrected to two reverse kernels;
VIR-6 surfaces 1 and 4 discharged/amended.

Gate: tests/test_chapter1_p0_corrections.py 12/12 PASS including mutants
M5/M6. Recorded that four of its arms require a reachable GPU fleet — a red
result there does not by itself indict the chapter.

Documentation only. No code, tests, config or manifests."

git push origin main && git push public main
```

---

## Hot State (Next Session Pickup)

**Where we left off:** Both chapters closed and verified; working tree has exactly three
documentation files modified/created and nothing else. Gate green 12/12 with the fleet up.
Nothing committed, nothing pushed, no pipeline run.

**Next action:** Michael reviews and dual-pushes (commands above). **One caveat worth knowing
before committing:** the closure statements assert `PASS` on a fleet-up gate run. If the rigs go
down again before the commit, the sentinel text stays true but the gate will not reproduce it —
Ch1 §17.2 states this explicitly, so the record is not misleading either way.

**Blockers:** None for the closure itself.

**Still open and NOT addressed by this pass (documentation-only brief):**
- Ch1 §17.3 — 14 items, notably the **manifest `parameter_bounds` vs §4.1 disagreement**
  (accepts `window_size=2`, which S172 excluded), **combined-session sampling can still select a
  non-certifiable mode**, D-1/D-2 unwired (hybrid certification blocked), and the four
  behavioural tickets (`run_with_config` writing `[]` silently; the TRSE `logger` `NameError`,
  still **`UNAVAILABLE`** — unverified at runtime; `min_workers` 1-vs-24).
- Ch2 §14.3 — F-3/F-4/F-5/F-7 and the six §12.1 inherited items. F-4 and F-5 are **confirmed and
  deliberately not repaired**; F-5 carries a live hazard note: **do not rename the rig hosts**,
  that would activate obsolete overrides.
- **No executable gate covers Chapter 2.** New this pass; for the gate owner.
- **S172 Phase 3** (`miner/range_miner_worker.py`) remains the next code deliverable per
  CLAUDE.md §6 — untouched by this session.

**Context needed first:** `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` §17 and
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §14 — the two closure statements are the authoritative
record of what is verified and what is still open.

---

*End of session — Chapter 1 & 2 closure, 2026-08-02*
