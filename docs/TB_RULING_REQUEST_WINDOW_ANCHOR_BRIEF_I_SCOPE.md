# TB RULING REQUEST — WINDOW-ANCHOR BRIEF I: THREE SCOPE ITEMS

**From:** Team Alpha · **Date:** 2026-08-21 · **Status:** submitted once, before §2.2 begins.
**Authority chain:** `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` → v1.0 review ruling →
`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` (`1bf49a5`, APPROVED) →
`TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` (design gate CLOSED, Brief I AUTHORIZED).
**Governing brief:** `docs/S172_WINDOW_ANCHOR_BRIEF_I.md`.
**Certified pre-change reference:** `gate12-passed-attempt9` = `e9ca800`. **Written against**
HEAD `205ae84`, digests verified, AC7 baseline captured before the first edit.

**State:** Brief I §2.1 (worker) is complete and verified. §2.2 (coordinator) has NOT begun and
will not begin until these are ruled. Nothing is committed. Line numbers below are given for
HEAD `205ae84` where the file is unmodified, and for the uncommitted working tree where §2.1
has already moved them; both are stated wherever they differ.

**This request asks for three rulings. It deliberately does not ask for a fourth** — §3B below
is stated for the record as work Alpha will do under either ruling, so that no item is held
pending a decision only part of it needs.

---

## ITEM 1 — C-3(b): `miner/range_miner_npz_writer.py:188` pulled into Brief I

**Requested ruling:** confirm (or refuse) that Brief I changes one line in a Phase-5 file.

Brief I §0 C-3(b) establishes, and Alpha has re-verified at source, that
`miner/range_miner_npz_writer.py:50-53` imports `_canonicalize_trial_context` directly from the
coordinator and `:187-191` declares its own `_CONTEXT_FIELDS` tuple containing `"offset"`. At
`:1026` that tuple drives a **required-key** comprehension,
`ctx = {k: metas[0][k] for k in _CONTEXT_FIELDS}`.

If §2.2 renames the coordinator's field to `window_anchor` and this tuple is left alone, Phase-5
assembly dies with `KeyError` on the first real trial. The tuple must therefore move in the same
commit as the coordinator, or the tree is red between Brief I and Brief II — which AC7 forbids
and which the "Brief II starts from the accepted Brief-I commit" lineage makes unavoidable.

**This is NOT the §4.5 NPZ generation-metadata work**, which stays in Brief II. No array is
added, removed, reordered, retyped or reshaped; the 22-array wall is untouched; no `savez` call
changes. It is a cross-phase consistency tuple.

**Alpha's position:** take the change, because it is the only option that leaves a green tree.
**Alpha is not deciding it unilaterally** — the brief itself directs that Beta rule at code
review, and Alpha would rather have the ruling before writing §2.2 than after.

---

## ITEM 2 — Chapter 2 §7.2 and its content gate document the defect Brief I repairs

**Requested ruling:** does Brief I update `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §7.2, the
F-4 finding rows, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` §3.1.2, and the two source assertions in
`tests/test_chapter2_content_gate.py`; or does that land as separate authorized work?

**Brief I does not mention this surface at all** — not in §3's firewall, not in §4's gates, not
in §6. Alpha found it when the post-§2.1 battery moved `test_chapter2_content_gate` from
**12/12 to 11/12**.

### What reds, and why it is not a broken test

`gate_source_anchors` (`tests/test_chapter2_content_gate.py:574-577`) asserts that §7.2's two
F-4 consumers are still present in live worker source:

```python
assert "def _offset_tail" in wsrc, \
    "§7.2's device-side offset consumer (_offset_tail) is gone from source"
assert "min(int(offset), n - window_size)" in wsrc, (
    "§7.2's host-side offset consumer (the residue-window slice) changed — "
    "the F-4 dual-consumer finding must be re-verified")
```

**Both now fail.** (The gate short-circuits on the first, so the battery log shows only one;
Alpha verified the second independently.) This gate is a **tripwire on the F-4 defect**, and
Brief I's entire purpose is to repair F-4. It is reporting exactly what it was built to report,
in its own words: *the finding must be re-verified.* The answer is that it is repaired.

**Alpha has not touched these assertions.** Loosening them would leave the gate green while no
longer proving what its message claims — the §2.44 pattern.

### The chapter text that is now stale

- `:801-805` — §7.2's anchored description of the fused path.
- `:1133` — the F-4 finding row: **"CONFIRMED. Settles Chapter 1 audit C-2 as an observed
  inconsistency — NOT the repair."**
- `:1346` — the closure finding table: **"CONFIRMED, not repaired."**
- `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:332,337` — the same coupling, described the same way.

### Two further weaknesses in the same gate, found while establishing the above

Alpha reports these because they bear on what any remediation must preserve, not as findings
Brief I proposes to fix.

**(a) The anchor loop checks resolution, not correctness.** `gate_source_anchors` resolves every
`file.py:line` citation in the chapter but tests only existence and range
(`if b > n or a < 1`). §2.1 grew `miner/range_miner_worker.py` from **2,629 to 2,892 lines**, so
every F-4 anchor still "resolves" while pointing at unrelated code, and the gate stays green on
all of them:

```
:648 -> 'seed_count: int'        :874 -> 'f"[0, {derived_max}] — dataset={path!r}, "'
:649 -> ''                       :196 -> '"a variant is declared both phase-capable and…"'
:694 -> 'provisioning fault…'    :197 -> ')'
```

**(b) `:578` cannot detect what its message claims.**
`assert "F-4" in text and "offset" in text, "§7 lost the F-4 offset finding"` is a keyword-presence
check over a 1,463-line document in which `"F-4"` occurs 5 times and `"offset"` 24 times. §7.2
could be rewritten, contradicted or emptied and it stays green off the finding table alone. The
suite has six mutants and **none targets this assertion**; `mutant_chapter_truncated` covers it
only incidentally.

**A constraint this places on any remediation:** because `:578` requires `"F-4"` to survive in
the chapter text, the finding may not be deleted. It must be **retained and re-disposed** —
which is the correct outcome regardless.

### Chapter 2 already named this commit as its own resolution point

The strongest argument is the chapter's, not Alpha's. `:1133` disposes F-4 as:

> **"Belongs in the future hybrid input-semantics design, not a standalone arithmetic patch."**

`:1346` repeats it, and `:831` states it a third time. **Brief I is that design.** The lineage
runs unbroken from that disposition: Chapter 2 F-4 → `AUDIT_STEP1_OFFSET_REACH.md` → the
sequencing ruling → proposal v1.0 → v1.1 → design gate CLOSED → this brief.

So updating §7.2 does not override the audited artifact. **It fulfils the disposition the
audited artifact wrote for itself**, at the commit that artifact pointed to. The question in
front of Beta is not whether Alpha may edit a closed chapter — it is whether the chapter's own
stated resolution point has now arrived.

**Alpha's position:** the chapter and the gate should move together with the repair, in Brief I,
so that no commit exists in which the chapter asserts a defect the code no longer has. But
Chapter 2 is audited, closed at `ef4b1c6` with content gate `09bbfbf`, and is on record as the
strongest of the three audited chapters. **Editing it is documentation authority Alpha does not
hold**, so Alpha will not touch it without this ruling.

---

## ITEM 3 — schema validation now precedes integrity validation in `ResidueResolver.resolve`

**This item splits. Only §3A needs a ruling.** §3B is fixture repair Alpha will perform under
either ruling and is stated here only so Beta does not hold the item for it.

### §3A — the ordering (RULING REQUESTED)

Brief I §2.1(f) directs that a payload carrying the retired `offset` key "fails loud *before*
any hashing or loading", and that `window_anchor` / `generator_phase` become required-key reads.
Alpha implemented both at the top of `resolve()`. The consequence is that a payload which is
**both** off-schema **and** carries a wrong `dataset_sha256` now reports the schema error rather
than the integrity error.

**The argument is containment, not preference.** Schema validation only ever *adds* a rejection:

- A **schema-valid** payload reaches the `dataset_sha256` check exactly as before, in the same
  place, with the same `ResidueVerificationError`. Blocker-6 Option C's requirement — that the
  check gate the method **before any cache return and before residue loading** — is untouched
  and was re-verified at source.
- A **schema-invalid** payload is rejected earlier than it used to be. It was already going to
  be rejected; only the diagnostic changes.
- Therefore **the set of accepted payloads after the change is a strict subset of the set
  before. No payload that was rejected is now accepted.**

**Failure classification is identical, verified at source.** `ResidueResolutionError` and
`ResidueVerificationError` both subclass `ResidueError` (HEAD `:512-521`; working tree
`:669-679`), and the worker routes that hierarchy to `_fail_stripe(..., retryable=False)`
(HEAD `:2000-2004`; working tree `:2258-2264`). Both orderings therefore produce the same
terminal disposition. **Only the diagnostic string a caller reads differs.**

**Scope of the effect: four certified gates, not one.** Alpha states this because a ruling
scoped to Gate 20 alone would leave three more in the same class:

| gate | what it pins |
|---|---|
| `test_s172_phase4_coordinator` Gate 20 | resolver end-to-end, **wrong** `dataset_sha256` |
| `test_s172_phase3_worker` Gate 15 | `[B6]` **missing** `dataset_sha256` → non-retryable |
| `test_s172_phase3_worker` Gate 16 | `[B6]` **mismatched** `dataset_sha256` → non-retryable |
| `test_s172_phase3_worker` Gate 17 | `[B6]` warm cache cannot bypass sha mismatch |

**Alpha's position:** the ordering is correct as implemented — the integrity of a payload that
is not on the schema is not a meaningful question — and the strict-subset property means it
cannot weaken Blocker-6. **Alpha requests confirmation rather than assuming it**, because the
change is contract-visible to any caller that reads the diagnostic, and because B6 is a Beta
blocker with its own certified gates.

### §3B — the Gate 20 / B6 fixture repair (NO RULING NEEDED; stated for the record)

Independently of §3A, these four gates no longer test what they were built to test. Gate 20
exists to prove **integrity** rejection; post-§2.1 its payload is also schema-incomplete, so it
never reaches the integrity path. Alpha will therefore, under either ruling:

1. give each fixture a valid `window_anchor` and `generator_phase`, so the gate isolates the
   integrity failure exactly as designed; and
2. add a **companion** assertion that a schema-incomplete payload raises the schema error — so
   the ordering becomes a tested property rather than an incidental one.

**Why this is not optional.** A Gate 20 made green by relaxing its assertion, or left green
without the repair, would be **green on a fact it does not check** — the same pattern Alpha
documented in Item 2(a) above. That is twice in one session, in two unrelated suites, and it is
the §2.44 failure mode the project has recorded six prior instances of.

---

## OBSERVATION — NOT A REQUEST: RC-1, the F1 claiming-model fixture staleness

**Independent of Brief I. True at `205ae84` before any edit, and equally true after.** Alpha
raises it because the AC7 baseline made its true scope visible and the record understates it.

Post-F1 (`c4e0037`), `assign_stripes` creates stripes `pending` / `claimed_by NULL`, and
`schedule_pending_stripes` is the only creator of a compute lease. Fixtures that drive
`assign_stripes` and then expect a claim or a dispatch fail **in the fixture**, not in
production. The shared signature is:

```
{'worker_id': None, 'attempt': 0, 'expected_substripes': None,
 'effective_cap': None, 'claimed': False}
```

**Eight suites share this mechanism**; skill §2.51 item 5 records it against **one**:

```
d1_engine 0/18 · d1_workflow 5/8 · d2 6/7 · d5 7/25 · d4_serial_backend 4/8
d6_production_adapter 0/9 · d6_threshold_path 4/17 · threshold_propagation 4/5
```

Alpha corrected its own first-pass baseline attribution here: four of these were initially
grouped under Gate D0-7 because a **nested** D0 run prints its `11/12` inside their logs. Their
own tallies are the figures above. The corrected grouping is recorded in
`logs/ac7_baseline_205ae84/ROOT_CAUSES.md` with the correction stated rather than silently
rewritten.

**Brief I does not propose to fix this and has not touched it.** Beta may wish to open it as its
own item; Alpha is not asking for that here.

---

## EVIDENCE

**AC7 baseline**, captured at `205ae84` **before the first edit**, 45 suites sequential,
`logs/ac7_baseline_205ae84/`: **32 green / 13 red**, 13 reds resolved to 6 root causes each with
a `file:line`. **Post-§2.1 battery**, same harness, `logs/ac7_post_21/`.

**§2.1 change scope:** `miner/range_miner_worker.py` only, +305/−42, one tracked file.
**44/44 `kernel_source` hashes identical to HEAD** (HEAD's `prng_registry.py` executed in-process
and compared entry by entry) — kernel ABI frozen byte-for-byte. Step-3 `continuation_phase`
(`full_scoring_worker.py:300`) untouched by construction. **13/13 §2.1 behavioural checks green.**

### AC1 is HALF satisfied at §2.1, and Alpha will not claim otherwise

AC1 requires **G-SEP-2 and G-SEP-3 green simultaneously** — the pairing is the whole of it.

- **G-SEP-2 precursor: GREEN.** A synthetic `generator_phase = 7` driven through the internal
  builder reaches position **17 of 17** on `lcg32_hybrid` as an `int32`, while every other arg
  position is byte-identical to the phase-0 case and the residue window is unchanged.
- **G-SEP-3: NOT SATISFIED, and not among the 13 checks.** Alpha states this plainly rather than
  presenting half of AC1 as AC1. Verified at source just now:
  `assert_generator_phase_supported("java_lcg", 7)` **returns cleanly** — the guard rejects
  nonzero phase only on the four no-phase variants, exactly as §2.1(d) specifies. It is not, and
  was never asked to be, the v1 pin.

**A related gap Alpha flags rather than resolves unilaterally:** a repo-wide search finds
`generator_phase` in **no `.py` file other than the worker**, so the v1 fail-closed pin against
nonzero production phase **currently exists nowhere**. Brief I §1 constraint 2 requires the pin
and constraint 9 requires the public schema stay fail-closed, but **§2 does not say which surface
enforces it.** §2.2's `build_stripe_assign_payload` gets `generator_phase: int` keyword-only with
no default — which makes the field mandatory, not pinned to zero.

Alpha's reading is that the pin belongs on the coordinator's public assign-payload validation in
§2.2, with G-SEP-3 written against it in §4, and Alpha will implement it there unless directed
otherwise. **This is not raised as a fourth ruling item** — it is a brief-specification gap Alpha
can close. It is stated here because it is adjacent to Item 3's validation-ordering territory,
and because if Beta wants the pin somewhere else, this is the cheapest moment to say so.

**Suites moved by §2.1 — four, all accounted for:**

| suite | base → post | disposition |
|---|---|---|
| `test_chapter2_content_gate` | 12/12 → 11/12 | **ITEM 2** |
| `test_s172_phase3_worker` | 17/17 → 9/17 | **8/8 fixture-shape only — zero assertion changes required** |
| `test_s172_phase4_coordinator` | 63/63 → 61/63 | Gate 20 = **ITEM 3**; Gate 23 = cascade of phase3 |
| `test_s172_phase5_d6_threshold_path` | 4/17 → 1/17 | payloads still emit legacy `offset`; resolves at §2.2 |

**Batteries are expected red between §2.1 and §2.2** — the worker now requires payload keys the
coordinator does not yet send. §2.1 is not a committable state alone.

**On the phase3_worker movement**, the largest single one: all eight failures are fixture-shape.
Two are `BuildContext.__init__() got an unexpected keyword argument 'offset'` (Gates 2, 14); six
are payloads carrying the retired key or missing a required one (Gates 7, 9, 12, 15, 16, 17).
**No gate's assertion was found to be testing the wrong thing, and Alpha proposes to change
none of them.** Alpha states this separately because a fixture update is mechanical whereas a
changed assertion is a claim that the gate was wrong, which would need justifying individually.

---

## WHAT ALPHA WILL DO ON EACH DISPOSITION

| item | if APPROVED | if REFUSED |
|---|---|---|
| 1 — npz_writer `:188` | change the one line in the Brief-I commit | **no green split exists — see below** |
| 2 — Chapter 2 / Ch 1 / gate | update chapter text, re-dispose F-4 as REPAIRED, re-point the two source assertions at `_generator_phase_tail` and the validated-domain code | **this is an AC7 amendment, not a deferral — see below** |
| 3A — ordering | proceed as implemented | invert the order so integrity precedes schema, and re-run the four gates |
| 3B — fixtures | proceeds regardless | proceeds regardless |

### Refusing Item 1 — the enumeration, so it is not mispriced as scheduling

Alpha's earlier phrasing ("Alpha would need Beta's sequencing") was too soft. Enumerated, the
splits are:

| split | outcome |
|---|---|
| worker only | **RED** — the worker requires payload keys the coordinator does not send. Measured, not predicted: this is the current tree state |
| worker + coordinator, npz in Brief II | **RED** — `KeyError` at `range_miner_npz_writer.py:1026`, the required-key comprehension over `_CONTEXT_FIELDS` |
| worker + coordinator + npz `:188` | **GREEN** — this is Item 1 approved |

**There is no fourth row.** The only way to make a worker-only Brief I green is to give
`window_anchor` / `generator_phase` defaults — which reinstates precisely the `.get(…, 0)`
class that constraint 4, G-REJECT-3 and mutant M2 exist to detect and kill.

Alpha states this to price the decision correctly, **not to pressure it**. Refusal is a choice
between a red tree across the Brief I/II boundary and reinstating the defect the brief was
written to extinguish. Both are Beta's to make; neither is a scheduling adjustment.

### Refusing Item 2 — this amends AC7, it does not defer

Alpha's earlier phrasing said refusal means carrying the red as "a declared, ruled red in the
AC7 report." **That was wrong, and Alpha withdraws it.** AC7 admits *pre-existing* reds,
identified as such **with evidence from `205ae84` before the changes**. `test_chapter2_content_gate`
was **12/12 green** in that baseline. Its 11/12 is caused by Brief I and has no home in AC7 as
written.

**Refusing Item 2 is therefore an amendment to AC7** — it requires Beta to admit a new class of
red: *caused by the change, ruled acceptable, deliberately left*. Alpha does not object to that
if Beta wants it, but it must be ruled explicitly rather than absorbed as the cheap option.
