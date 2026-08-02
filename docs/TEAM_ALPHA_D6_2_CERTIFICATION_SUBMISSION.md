# TEAM ALPHA → TEAM BETA — D6.2 submitted for certification

**Committed and dual-pushed at `f7583bc`.** 11 files, 6,115 insertions. Nothing was committed from
the sandbox; WATCHER was not run.

**Result: 29/29 gates · 377 assertions · 23/23 mutants killed · PASS.**

`utils/checkpoint_d6_2.py` (schema, both digest layers, CSR sessions, run-id grammar, path
confinement, `run_context_digest`, nine-row recovery, reconciliation, write transaction) and
`tests/test_s172_d6_2_checkpoint_reconciliation.py`.

**The flush now writes the complete 24-field canonical state, `_FLUSH_CLEAR_IN_MEMORY = True`, and
the finalizer is fed the reconstructed cumulative state rather than the truncated stump. The S166
OOM protection is real for the first time.**

Full evidence: `docs/S172_D6_2_IMPLEMENTATION_REPORT.md`.

---

## 1. Against the brief and the binding addendum

- **§2.2 — zero drift.** Dtypes derived from `CANONICAL_ARRAY_CONTRACT`, checked against a
  hand-transcribed oracle.
- **Addendum §1 — settled as ruled.** Identity fields **are** in `member_content_digest`; it
  excludes only itself; computed **last**; **fixed order** — a reversed mapping yields the
  identical digest. The state preimage emits the addendum's physical order exactly, `prng_base`
  absent. **Ten permutations → one digest.**
- **Addendum §2 — all nine recovery rows**, row 5 included as its own case, nine distinct outcome
  labels asserted reached.
- **§4.2 — all three resume hops**, driven through the **real** WATCHER filter, with an undeclared
  control key proving the filter still filters.
- **Addendum §4 — both trial-namespace checks**, with a real `enqueue_trial` producing a genuine
  `WAITING` trial 0 that the live pre-flight rejects.

**Two real defects the gates caught mid-build**, both fixed before green:
1. a **forged sequence** in a digest-failing member A could have raised the next sequence —
   violating §4.6. First implementation used `max(seq_b, probe_sequence) + 1` on row 2, so **one
   flipped byte in A could have pushed the run's numbering anywhere.**
2. an **unreadable A** raised a non-`CheckpointError` instead of falling to row 1.

---

## 2. Four items requiring Beta's ruling — Alpha's positions

### 2.1 ⚠ `trial.number` is NOT the number that lands in the record — the addendum guarded the wrong counter

**Verified at source.** `trial_counter = {'count': 0}`
(`window_optimizer_integration_final.py:2361`) → `+= 1` (`:2382`) →
`trial_number=trial_counter['count']` (`:2399`). **A process-local, 1-based ordinal that restarts
every run.** `optuna_trial.number` is study-scoped, 0-based, and reaches only partition routing
(`:2384`) and `result.iteration`.

`trial_number` is part of the replay key `(seed, trial_number, skip_mode)`. **So addendum §4's two
checks, alone, cannot close the collision they exist to prevent** — a resumed study continues
Optuna's numbering while the record ordinal restarts at 1 and collides with recovered trial 1 under
different canonical contents.

**Claude Code implemented both checks exactly as specified** — check 1 remains the only thing that
catches an enqueued trial — **and additionally continued the record ordinal from the recovered
maximum**, so the first new trial is `recovered_max + 1`. **No Optuna number is read, written or
shifted**, and the optimizer execution cursor is still not restored. `G-TRIAL-NAMESPACE` pins it.

**Alpha's position: accept.** The ordinal continuation is what actually closes the hole. **Alpha
does not recommend converting the record's `trial_number` to `trial.number`** — that touches
D1/D3.25 record semantics and is outside D6.2.

**This is submitted as a correction to the addendum, not as a deviation from it.**

### 2.2 Four non-regression suites had to change — REV5 §10 is unresolvable as written

**REV5 §10 requires D6.1 green, while REV5 §1/§3.3/§8 require the three facts D6.1 pins to become
false.** **That is Alpha's defect** — §10 was written without checking what D6.1 asserts.

**Resolution taken: port, never relax.** Every changed assertion is itemised in report §13.1.
Highlights:
- `_FLUSH_CLEAR_IN_MEMORY is False` → `is True`, with the reason;
- `"four-field" in _CHECKPOINT_SCHEMA_VERSION` → `not in`, **plus** an assertion the version is
  *imported* rather than restated;
- `_flush_inspect_pair` / `_PAIR_*` → a thin helper classifying through the **live**
  `recover_checkpoint`, **reimplementing no decision**. Beta's transaction-identity counterexample
  (identical seed sets, changed score, mixed pair) is still constructed and still must not classify
  as consistent;
- **G-COMMENT-TRUTH: four now-false D6.1 phrases retired and asserted ABSENT**, six required phrases
  retained;
- **M8 relocated, not deleted** — its subject moved modules, so its mutant moved to the D6.2 suite
  and the slot became "the fail-closed run-context guard removed".

**Requires Beta's ratification** — §10 was in an approved brief.

### 2.3 Member A's seed column is `seed`, not `seeds`

Addendum §1 says *"`seeds` and `score`"*; REV5 §2.1 says the checkpoint stores **record** field
names and *"do not apply that rename here."* **Alpha's ruling: `seed` is correct.** The addendum was
identifying *which two fields*, not reopening the naming domain; the loose wording was Alpha's,
carried from D6.1's array name.

### 2.4 `member_role` added to the identity block beyond §3.3's table

**Alpha's ruling: accept.** Every recovery row turns on "is this A or B", and deriving that from a
filename is the naming-convention-instead-of-runtime-check pattern `_flush_assert_not_alias` exists
to prevent. Declared with reasoning in report §2.4.

---

## 3. Two further declarations — Alpha judges both correct, no ruling sought

- **Combination-matrix row 4 is a rejection**, because the reconstruct/finalize-only surface REV5
  §4.4 conditions on does not exist in this entrypoint. **This is the specified behaviour** —
  §4.4 reads *"may reconstruct/finalize… **if that surface exists**; otherwise **reject before
  optimization with a specific error**."*
- **The trial-namespace floor travels on an attribute seam** (`strategy._resume_trial_floor`), the
  pattern S149/S152 already established for `_survivor_accumulator`, rather than a new
  `OPTIMIZE_FORWARDED_KWARGS` entry. **Alpha verified the justification at source:**
  `OPTIMIZE_FORWARDED_KWARGS` (`window_optimizer.py:334`) is AST-pinned against the live signature
  (`tests/test_chapter1_p0_corrections.py:234`) and is what `strategy_contract_gap` (`:520-529`)
  measures the three gated strategies against — **widening it would change an unrelated contract.**
  The floor is **read and enforced, not advisory**, and gated end to end.

---

## 4. Non-regression

**D6.2 29/29 · D6.1 (ported) 15/15 · D6 3.A 9/9 · D6-threshold 17/17 · D3.5 60/60 · D3.25 13/13 ·
D3 10/10 · D3.0 10/10 · D0 12/12 · D4 8/8 · import gate 7/7 · Phase 1 6/6 · Phase 2 6/6 ·
Phase 3 17/17 · 6-P0.5 37/37 · Ch1 P0 12/12 · `test_prng_encoding` 8/8.**

**D3.25 and D3.5 unchanged**, as REV5 §12 requires.

The five reds in the pre-commit run were **one cause** — Phase-4 Gate 22 building `changed_py` from
`git status --porcelain`, which sees untracked files. **Proved by isolation before the commit, and
by resolution after it: Phase 4 is 63/63 at `f7583bc`.** Gate 22 was not edited.

**No Wall A/B rerun** — Beta confirmed none required.

---

## 5. Ruling requested

1. **§2.1** — accept the record-ordinal continuation, and note the addendum's §4 targeted a
   different quantity than the replay key uses.
2. **§2.2** — ratify the four ported suites, and the finding that REV5 §10 was unsatisfiable
   alongside §1/§3.3/§8.
3. **§2.3, §2.4** — confirm.
4. **Certify D6.2**, which is the last standing Phase-7 blocker. Beta has already ruled that the
   multi-stripe protocol item does not block the soak.

## 6. VIR declaration

**Execution proof:** every gate prints its name and a non-trivial assertion count (377 total); the
parity gate reports compared artifact digests; recovery and combination gates name the row under
test. **Clean control:** row 6 is the uninterrupted reference; a normal fresh run passes with both
resume controls empty; every mutant carries a positive control requiring its detector to pass
against unmutated source. **Fault injection:** 23 mutants, four-part kill rule; **three re-credited
where the nominated gate could not see the defect, documented inline.** **Sentinel:** `PASS`.
**Unavailable-observer:** D6.2 has **no fleet dependency** — nothing contacts a rig, GPU or
coordinator; **no arm is `UNAVAILABLE`.** **Audit scope:** repo-scoped at `f7583bc`.
**Unavailable surfaces:** host state on VM101 and the rigs; the live `KERNEL_REGISTRY` if changed;
deployed uncommitted files outside the tree.
