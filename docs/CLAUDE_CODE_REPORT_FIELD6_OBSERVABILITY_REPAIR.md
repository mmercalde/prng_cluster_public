# CLAUDE CODE REPORT — FIELD-6 OBSERVABILITY REPAIR

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md`
**Authority:** `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (`d391a5c`) — Field 6 ruled
**UNOBSERVED, instrumentation-output defect**; sequencing item 3.
**Host:** VM101 `192.168.3.177`, user `michael`, repo `/home/michael/distributed_prng_analysis`,
venv `~/venvs/torch`. **Tree at start: `73633e7`, clean of tracked modifications.**
**NOTHING COMMITTED, NOTHING PUSHED.**

---

## 0. The falsifiable question, answered

> After this repair, does the trial-terminal `[S172-BP] summary` record carry
> `deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water` such that
> (a) the emitted values vary appropriately across two different driven pump populations, and
> (b) a run in which no pump pass occurred emits the literal `UNOBSERVED` for both — proven by
> a gate, not by key presence?

**YES, on both limbs, proven by `G-FIELD6` (three arms) with `G-MUT-FIELD6` (three mutants, all
DETECTED).** The values are asserted EXACTLY against a relation **derived from `_pump_deferred`
as read live**, not transcribed from its docstring, and both arms read the values out of the
**emitted log line**, never out of the returned dict.

Measured, final state:

```
OBSERVED, K=3 distinct attempts x 4 frames:
 ... admission_dispositions_max_per_iteration=0 deferred_distinct_attempts_high_water=3 pump_liveness_probes_high_water=8

OBSERVED, K=6 distinct attempts x 4 frames:
 ... admission_dispositions_max_per_iteration=0 deferred_distinct_attempts_high_water=6 pump_liveness_probes_high_water=14

NO PUMP PASS AT ALL:
 ... admission_dispositions_max_per_iteration=0 deferred_distinct_attempts_high_water=UNOBSERVED pump_liveness_probes_high_water=UNOBSERVED
   returned dict: None / None
```

Note the population was **not** what moved between the two observed runs in the way that matters:
`deferred_high_water` reports `12` then `24`, i.e. the population — which is exactly R-1's point
that it **cannot test the guarantee at all**. `K` is what the falsifier fields track.

---

## 1. Files changed — THE `git add` LIST

Build the stage list from **this section**, never from recall. Never `git add -a`.

```
 M miner/range_miner_coordinator.py            scope A + B + C + F (3 definitions)
 M tests/test_s172_staging_backpressure.py     scope D + E (the new gate, its mutants)
 M tests/test_s172_r1_drain_remedy.py          §5 FORCED — outside the brief's enumerated scope
 M tests/test_s172_mp1_drain_attribution.py    §5 FORCED — outside the brief's enumerated scope
 M tests/test_s172_attempt6_remediation.py     §5 FORCED — outside the brief's enumerated scope
 A docs/CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md   this file
```

**§5 is the section Beta must read first.** Three certified suites were edited. None of the
three is in the brief's permitted-change list; each is a mechanical consequence of Scope C; none
weakens a gate. The alternative — leaving them red — is stated there with its cost.

Untracked paths present in the tree and **not** touched by this work: the four
`PIECE_MATCHER_*.md`, `piece_matcher/`, `docs/SESSION_CHANGELOG_20260819_S1.md`.

---

## 2. The change

### A — the UNOBSERVED sentinel (`_bp` init, `range_miner_coordinator.py`)

`deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water` now seed to
**`None`**, not `0`, with the rationale recorded beside the existing R-1 falsifier comment. The
dict keeps `None` (JSON-safe null); only the log line substitutes the literal.

### B — the None-aware update, and the trap

The former update did `int(self._bp[...])` **before** `max()`. Against a `None` seed that raises
`TypeError`, and the blanket `except Exception: pass` immediately above swallows it — so both
fields would have stayed `None` **forever** and every future run would have falsely reported
UNOBSERVED. This is not hypothetical: **mutant M3 reproduces it exactly**, and §4 records the
resulting line.

The update is now `observation if current is None else max(int(current), observation)`, still
under `_bp_lock`, still wrapped, still **INLINE** — no new `def` in this module (§3).

**Correction to the brief, Scope B's closing note.** The brief says *"a pump pass over an empty
`_deferred` legitimately records `0`."* **It does not, and cannot.** `_pump_deferred` early-
`return`s inside `with self._admission_lock` when `not self._deferred`, so the instrument is
**never reached** with an empty store — verified, §6 edge case 2. Every pass that does reach it
carries `len(seen_keys) >= 1` and `probes >= 1`. A recorded `0` is therefore **unreachable from
this call site**. The None-aware form still does not special-case zero away — if a future change
moved that early return, a genuine `0` would record as `0` and remain distinct from `None`.

### C — the emitter (**this was the defect**)

Both keys appended to the **END** of the existing `[S172-BP] summary` format string, `%s`,
additive, same grep-stable line — the `[ATTEMPT-6] additive series` precedent directly above.
Integer when observed; the literal `UNOBSERVED` when `None` (the `staging_jobs_per_sec=n/a`
precedent for non-numeric emission in this same record). The dict↔line mapping is documented in
the method docstring and at the call site.

**Substring-collision check, re-performed on the live emitted lines:** a grep for
`deferred_high_water=` matches **1** token on both observed lines and **1** on the UNOBSERVED
line — `deferred_distinct_attempts_high_water=` does not contain it. Existing greps are safe.
**Live extractors: none exist.** `grep -rn 'S172-BP' scripts/` returns nothing, and the only
non-test reader of this line in the repo is the coordinator itself.

### D — the completeness-gate key list

Both keys added to `gate_metrics_are_grep_stable_and_complete`'s metrics-dict list **and** to its
summary-line substring list. Key presence is satisfied by `key=UNOBSERVED`, which is precisely
why that gate is **not** the evidence — `G-FIELD6` is.

### F — the rider: the `:7741` docstring debt (pre-authorized, non-executable)

`SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` §3.1 item 1 recorded the debt and its rule:
ride the first commit that legitimately touches this file. This commit does. Replaced verbatim
with the report's recommended wording:

```diff
-        recorded, so every entry that is DROPPED is dropped on its own fresh,
-        under-lock `_attempt_live_locked` call — byte-identical to the old loop.
+        recorded, so no drop ever rests on a REUSED observation. R-3's
+        end-of-pass sweep deliberately retires every retained frame of a key on
+        ONE fresh negative probe; what holds per-entry is that the observation
+        behind it is never reused.
```

**Docstring-only. No behaviour, no digest-bearing logic** (the AST proof blanks docstrings, so
this contributes nothing to §3's changed set on its own). **Entered under the debt rule, not as
scope creep** — flagged here so Beta sees which rule admitted it.

---

## 3. AST scope proof — the added-definition set is UNCHANGED

Per-definition SHA-256 with docstrings blanked, `HEAD:miner/range_miner_coordinator.py` vs live:

```
ADDED   : (none)
REMOVED : (none)
CHANGED : RangeMinerCoordinator.__init__
          RangeMinerCoordinator._pump_deferred
          RangeMinerCoordinator.log_staging_backpressure_summary
total definitions HEAD/live: 289 / 289
```

**No new `def` in `range_miner_coordinator.py`** — MP-1's certified `gate_e2_ast_scope_proof`
asserts the added set exactly, and it is empty. Three definitions changed, one per scope item
(A→`__init__`, B+F→`_pump_deferred`, C→`log_staging_backpressure_summary`).

`log_staging_backpressure_summary` is the definition that forces §5.

---

## 4. Per-arm gate output and mutation evidence

### `G-FIELD6` — PASS (three arms, one gate)

| arm | subject | what it asserts |
|---|---|---|
| **1** population variance | two runs, `K=3` and `K=6`, frames fixed at 4 | both keys present ON THE EMITTED LINE and integral; `distinct == K` exactly (3, 6); `distinct(K2) > distinct(K1)`; the **derived** probe identity `probes == frames + 2*(K-1)` at both endpoints (8, 14); `probes(K2) > probes(K1)`; `distinct != probes` on both runs; `deferred_high_water == K*frames` (12, 24) so the population is shown to be the thing that did **not** distinguish them |
| **2** UNOBSERVED pin | a coordinator over which no pump pass ran | the line carries `deferred_distinct_attempts_high_water=UNOBSERVED pump_liveness_probes_high_water=UNOBSERVED`, **adjacent at the end of the line**, and the returned dict carries `None` for both |
| **3** dict↔line coherence | the same two observed runs | `int(line[key]) == int(dict[key])` for both keys on both runs |

**The probe relation was DERIVED from `_pump_deferred` as read, not transcribed.** With frames
grouped by attempt and every staging slot held:

* the first attempt is admitted, and **R-2 discards its key from `live_keys` at every grant**
  (`live_keys.discard(_key)` immediately after `_try_admit_locked` returns True), so **each of
  its `frames` entries costs one fresh probe**;
* each of the other `K-1` attempts probes **once** in the main scan — the memo covers its
  remaining frames — and is refused admission (`_admitted` already holds a different key), so it
  is **retained** in `live_keys`;
* **R-3's end-of-pass sweep** re-probes each of those `K-1` retained keys exactly once
  (`probes += len(live_keys)`).

⇒ `probes = frames + (K-1) + (K-1) = frames + 2*(K-1)`. Independently corroborated: this is the
same identity `gate_g8e_the_falsifier_fields_are_measuring` measures at `K=4/frames=50` (56) and
`K=20/frames=10` (48), and the same one `_pump_deferred`'s docstring states as the G8b identity.

**Fixture honesty.** The population is built through the **real `enqueue_staging`** with every
staging slot genuinely held on the **same semaphore `enqueue_staging` acquires** — the G4 idiom —
so every frame takes the production `action == "deferred"` branch. The helper asserts
`len(_deferred) == K*frames` **before** the pump and again **after** it, so a fixture that
silently fail-fasted or back-pressured cannot masquerade as a measurement.

**One worker per attempt, and not for convenience.** F1's `claim_stripe` raises
`LeaseInvariantError` on a second concurrent compute claim by one worker, so `K` concurrently
deferred attempts **is** `K` distinct workers by construction — which is exactly why R-1 argues
the distinct-attempt count is bounded by the frozen cohort. The first draft of this fixture used
one worker and was refused by production; that refusal is recorded here rather than worked
around.

### `G-MUT-FIELD6` — PASS (three mutants, three DETECTED, zero SURVIVED)

Each mutant is compiled from **live production source** via `ast.parse` + `exec` against
**`miner.range_miner_coordinator.__dict__`** as globals — the A8-B2 lesson: a verbatim copy
exec'd in the test module's globals resolves its callees there and escapes the mutation. Each
`.replace()` anchor is asserted to have actually matched (`assert mutated != original, "was NOT
APPLIED — the anchor moved"`), so an anchor drifting under a future edit reports as a *harness
failure*, never as a silently un-applied mutant.

| mutant | APPLIED | EXECUTED | DETECTED by | detecting assertion, verbatim |
|---|---|---|---|---|
| **M1** emitter hardcodes `0` for both values | anchor matched | emitter ran, line emitted | G-FIELD6 **arm 1** | `distinct attempts on the K1 run: 0 != 3` |
| **M2** emitter transposes the two arguments | anchor matched | emitter ran, line emitted | G-FIELD6 **arm 1** (arm 3 also) | `distinct attempts on the K1 run: 8 != 3` |
| **M3** update path restores the `int()` cast over the `None` sentinel | anchor matched | **proven separately**, see below | G-FIELD6 **arms 1 and 3** | `[K1] deferred_distinct_attempts_high_water='UNOBSERVED' is not an integer on an OBSERVED run: [S172-BP] summary … deferred_distinct_attempts_high_water=UNOBSERVED pump_liveness_probes_high_water=UNOBSERVED` |

**M2 detectability is enumerated, not assumed.** A transposition is undetectable at any
population where `distinct == probes`. `G-FIELD6` therefore **asserts** `distinct != probes` on
both runs (`3 != 8`, `6 != 14`) before relying on the transposition being visible — the property
is a gate assertion, not a claim in prose. At `K=1` the two would be equal only when
`frames = 1 + 2*(1-1) = 1`; both scenarios are far from that.

**M3 carries its own EXECUTION PROOF, independent of the gate's verdict.** Before running
`G-FIELD6` under the mutant, the gate drives a real observed run under the mutated
`_pump_deferred` and asserts **both fields are still `None`** — i.e. the `TypeError` really did
fire and really was swallowed. Without that arm, "the gate reddened" would not distinguish "the
mutant did the harmful thing" from "the mutant broke on import". This mutant exists because it is
**the exact silent-failure mode Scope B warns about**, and the M3 row above is what that failure
looks like in the persisted artifact: an OBSERVED run reporting UNOBSERVED.

---

## 5. ⚠ THREE CERTIFIED SUITES EDITED — OUTSIDE THE BRIEF'S ENUMERATED SCOPE

**The brief did not anticipate these, and I am not treating that as authorization.** All three
are mechanically forced by Scope C — the mandated emitter change — and none of them can be
avoided while Scope C stands. Each is reported with what it was, what it now is, and why it is
not a weakening. **If Beta prefers them reverted, the cost is stated per row.**

### 5.1 `tests/test_s172_r1_drain_remedy.py` — TWO forced changes

**(a) `DECLARED_CHANGED` grew by one entry.** `gate_scope_proof` compares **live** source against
R-1's pinned anchor `c403a37` and asserts `changed == DECLARED_CHANGED` **exactly**. Scope C
changes `log_staging_backpressure_summary`, which R-1 did not declare. Measured red:

```
[FAIL] AST scope proof   undeclared changes:
       ['RangeMinerCoordinator.log_staging_backpressure_summary']; declared but unchanged: []
```

Added with a comment naming **FIELD-6, not R-1**, as the owner of that entry, so the declaration
does not misattribute the change. **This does not weaken the gate** — it still asserts exactness,
and it is now self-protecting in the other direction: if a future session reverts the emitter,
the proof reds with *"declared but unchanged"*.

*Structural note Beta may want to rule on separately:* this shape means **every** future
authorized commit touching `range_miner_coordinator.py` must grow this list or the proof reds
permanently. It has been invisible so far only by coincidence — R-1 changed exactly the two
definitions MP-1 had already declared (`_pump_deferred`, `__init__`), so MP-1's proof absorbed
R-1 for free. Field 6 is the first commit for which that coincidence does not hold.

**(b) the M9 mutant fixture (`_metric_mutant`) now tracks production's None-aware update.**
`_metric_mutant` is a hand-copied `_pump_deferred` whose `_bp` write still did
`int(self._bp[...])` — the M3 shape — with **no `try/except` around it**. Against the `None` seed
it raises `TypeError` and the mutant **dies before it can be measured**. Measured red:

```
[FAILED] M9  wrong falsifier field -> G8e:
         TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

That is a **fixture failure, not a surviving mutant** — worse than a red, because it looks like
one. The mutation actually under test — `record(n_deferred, seen_keys, probes)` substituting the
population for the distinct count — is **untouched**; only the four lines that copy production's
`_bp` write were made None-aware, exactly as production is.

**Result: 44/44** (from 42/44).

### 5.2 `tests/test_s172_mp1_drain_attribution.py` — one forced change

Identical cause to 5.1(a): `gate_e2_ast_scope_proof` asserts `changed == DECLARED_CHANGED` against
MP-1's pinned anchor `2c38f8c`. `log_staging_backpressure_summary` added, commented as
**FIELD-6's, not MP-1's**. **Result: 38/38** (from 37/38).

### 5.3 `tests/test_s172_attempt6_remediation.py` — one forced change, and it is a JUDGEMENT CALL

`fair4_arm0` runs the back-pressure battery as a subprocess and asserted the **transcribed tally**
`"50/50 checks green"`. Scope E legitimately grows that suite to 52. Measured red:

```
[FAIL] FAIR-4/0  the full S172-BP battery: the battery did not report 50/50
```

**`50 -> 52` is the amendment Beta explicitly called wrong at R4-1** (*"same brittle shape, fails
on the next legitimate change"*), so it is not the one taken. Instead the pin is replaced by the
two properties it was standing in for, **neither of which a growing suite can trip**:

* the suite's **own COMPLETION SENTINEL**, printed only when `passed == total`; plus a parsed
  `passed == total` assertion — both already implied by the `returncode == 0` assertion one line
  above, since `main()` returns `0` **only** on a full pass;
* a **FLOOR**, `total >= 50`, so gates being **deleted** still reds this arm. That is the one
  thing the transcribed tally caught that a bare "green" does not, and it is preserved.

A stale `(50/50)` in `fair3_arm1`'s docstring was corrected to `(>= 50, and green)` in the same
edit. **Result: 78/78** (from 77/78 at baseline; see §7 on that baseline red).

**This is the one change in the diff where I exercised judgement rather than mechanism.** The
mechanical alternative was `50 -> 52`, which Beta has already ruled against, and the do-nothing
alternative leaves a certified suite red. If Beta disagrees, the row to revert is this one.

---

## 6. Edge cases — behaviour enumerated for EVERY case in the input space (self-check #14)

Each row was **executed on VM101 against the final tree**, not reasoned about.

| # | case | behaviour | evidence |
|---|---|---|---|
| **1** | **No pump pass at all** | both `None` in the dict, both `UNOBSERVED` on the line | `G-FIELD6` arm 2; line reproduced in §0 |
| **2** | **Pump pass over an EMPTY `_deferred`** | **the instrument is NOT reached.** `_pump_deferred` early-`return`s inside `with self._admission_lock` when `not self._deferred`, above the update block. Both fields **retain their prior value**, which on a fresh coordinator is `None`/`UNOBSERVED` | executed: fresh coordinator, `_deferred == []`, `_pump_deferred()` → `None` / `None`. **This contradicts the brief's Scope-B note** (§2 B) |
| **3** | **Pump passes, then the trial ABORTS** | the last high-waters are emitted. `serve_trial` has exactly **one** top-level `Return` (`:10665`), immediately preceded by `log_staging_backpressure_summary` (`:10660`), *outside* the big `try` — so every path that leaves the serve loop normally emits, committed or aborted | AST of `serve_trial`: one top-level `Return`; the other five `Return`s are in nested closures |
| **3b** | **the qualification the call-site comment does not carry** | the summary is **NOT** emitted when `serve_trial` itself RAISES. Its `Try (9631-10642)` has **zero except handlers**, only a `finally`, so `raise primary` at `:10524` (threshold-provenance violation) and the two pre-loop config `ValueError`s at `:9362`/`:9420` propagate past `:10660`. **Pre-existing, unchanged by this commit**, and in all three cases nothing at all is emitted — no fabricated value is produced | AST: `handlers: []` on the serve-loop `Try` |
| **4** | **Exception inside the update block** | fields **retain their prior value** (possibly `None`) — truthful degradation, never a fabricated `0` | executed: after a good pass the fields read `3 / 8`; a pass with `_bp_lock` replaced by a raising context manager leaves them at **`3 / 8`** |
| **5** | **Two runs in ONE coordinator lifetime** | **`_bp` is NEVER reset.** The only per-trial write is `self._bp["trial_started_at"] = start` (`:9520`); no code path clears or re-seeds the dict. Both fields are therefore **coordinator-lifetime high-waters**, not per-trial ones | executed: trial 1 at `K=6` → `6 / 14`; trial 2 at `K=2` in the same coordinator → still **`6 / 14`** |

**On row 5, the meaning across the boundary, stated rather than left to inference.** Under the
S172 certification boundary (§2.24: *one active range-miner trial per coordinator process*) a
second trial in one coordinator process is **not a certified production shape**, so in production
the lifetime high-water and the trial high-water are the same number. Outside that boundary they
are not, and the emitted value would be **the maximum over every trial the process has served** —
a lower bound on the true resident maximum for the *run*, and an over-statement for the *trial*.
This is pre-existing behaviour shared by every `_bp` counter (`deferred_high_water`,
`pause_events`, `staging_jobs_completed` …); this commit neither creates nor repairs it, and
**no per-trial reset was added** — that would be an out-of-scope behavioural change.

---

## 7. Runs — final state, sequential

Every suite was run **after the last change**, `python3 -u <suite> | tee`, **never piped to
`tail`**, **sequentially** (concurrent S172 runs flake Part B's free-space arm), venv
`~/venvs/torch` active, on VM101.

A **BASELINE run of all 20 suites was taken on the unmodified tree at `73633e7` first**, so every
red below is attributed by differential rather than by assertion.

| suite | baseline (`73633e7`) | final | verdict |
|---|---|---|---|
| **`test_s172_staging_backpressure`** | 50/50 | **52/52** | **+2: `G-FIELD6`, `G-MUT-FIELD6`** |
| `test_s172_r1_drain_remedy` | 44/44 | **44/44** | green — via §5.1 |
| `test_s172_mp1_drain_attribution` | 38/38 | **38/38** | green — via §5.2 |
| `test_s172_attempt6_remediation` | **77/78** | **78/78** | green — via §5.3; see below |
| `test_s172_h1h2_instrumentation` | 62/62 | 62/62 | unchanged |
| `test_s172_staging_partb` | 24/24 | 24/24 | unchanged |
| `test_s172_f1_lease_origin` | 18/18 | 18/18 | unchanged |
| `test_s172_f1_f2_active_lease` | 16/16 | 16/16 | unchanged |
| `test_s172_admission_liveness` | 16/16 | 16/16 | unchanged |
| `test_s172_defect_a_transport_recovery` | 29/29 | 29/29 | unchanged |
| `test_s172_resolved_execution_set` | 34/34 | 34/34 | unchanged |
| `test_s172_elapsed_roundtrip` | 6/6 | 6/6 | unchanged |
| `test_s172_phase3_worker` | 17/17 | 17/17 | unchanged |
| `test_s172_phase2_protocol` | 6/6 | 6/6 | unchanged |
| `test_s172_phase1_scaffolding` | 6/6 | 6/6 | unchanged |
| `test_gate12_gpu_gate` | 9/9 | 9/9 | unchanged |
| `test_gate12_cleantree_admission` | 31/31 | 31/31 | unchanged |
| **`test_s172_phase4_coordinator`** | **63/63** | **62/63** | **Gate 22 only — see below** |
| `test_s172_admission_binding` | **11/20** | **11/20** | **PRE-EXISTING, zero differential** |
| `test_s172_phase5_d6_production_adapter` | **0/9** | **0/9** | **PRE-EXISTING, zero differential** |

Suite set = the S181 §5 thirteen-suite regression battery **∪** the seven suites the R-1..R-4
drain-remedy commit ran (`SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` §2) — the union,
because this commit touches `_pump_deferred` and `__init__`, the two definitions R-1's and MP-1's
gates pin.

### Pre-existing reds, with evidence

**`test_s172_admission_binding` 11/20 — PRE-EXISTING, zero differential.** The nine-gate failure
set is **byte-identical** between baseline and final once run-scoped ids (`set_id`, `run-…`) are
normalised. Matches the standing record (`SESSION_CHANGELOG_20260814_S181.md` §5, skill §7).

**`test_s172_phase5_d6_production_adapter` 0/9 — PRE-EXISTING, zero differential.** Failure set
byte-identical between baseline and final. Red at the base since F1; stale fixture, not a
production defect (`SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` §2). **Not chased.**

**`test_s172_attempt6_remediation`: the BASELINE red was `FAIR-6/4`, not `RXP-1/4`.** The brief
names RXP-1/4 as the disclosed flake; the baseline run on the **unmodified** tree instead reddened
`FAIR-6/4` (*"an admission turn contributed 0.433s, more than D_adm + one overrun registration"*)
— the same non-deterministic timing class, a different arm. It did **not** recur on either
post-change run. Reported as observed, nothing more.

### The one NEW red, and it is the documented one

**`test_s172_phase4_coordinator` — Gate 22, and ONLY Gate 22:**

```
[FAIL] Gate 22: coexistence (use_range_miner, PWC/ZMQ): unexpected changed .py files:
       {'tests/test_s172_mp1_drain_attribution.py',
        'tests/test_s172_attempt6_remediation.py',
        'tests/test_s172_r1_drain_remedy.py'}
```

The detector builds `changed_py` from `git status --porcelain`, so a **modified tracked**
non-allowlisted `.py` trips it exactly as an untracked one does — the correction Claude Code made
to Alpha's brief and Beta ratified (§2.33: *"committed once ⇒ clears forever" is false*).

It names **exactly the three §5 suites and nothing else**:
`miner/range_miner_coordinator.py` and `tests/test_s172_staging_backpressure.py` are both already
on the allowlist, so the brief's own two scoped files are clean.

**The allowlist was NOT widened, and must not be.** The answer is *"commit the files"* — Gate 22
self-clears on a clean committed tree. This is the **seventh** recorded occurrence.

*Trade-off, stated so Beta can price it:* had the three §5 suites been left untouched, phase-4
would read 63/63 and `r1`/`mp1` would read 42/44 and 37/38 with `COMPLETION SENTINEL: FAIL`. The
Gate-22 red self-clears at commit; the other two would not.

---

## 8. Prohibitions — nothing forbidden was touched

| prohibition | status |
|---|---|
| No admission / lease / pump / queue / cursor / publication / scheduler **logic** change | **HELD.** The only behavioural delta is sentinel init + None-aware update + emission. `_pump_deferred`'s iteration order, selection policy, `_try_admit_locked` call, nonblocking slot acquire, `still`/`released` bookkeeping, R-2 discard, R-3 sweep and the `finally` resume are byte-identical |
| No new performance optimization | **HELD** — none present |
| No heartbeat-disposition work; no missing-expiry-summary work | **HELD** — neither touched; both remain tracked separately |
| No new `def` in `range_miner_coordinator.py` | **HELD** — §3, added set empty |
| No Gate-12 rerun, no production launch, no touching `gate12_launch.sh` or fleet scripts | **HELD** — `scripts/` byte-unchanged; nothing launched; no GPU work; no rig SSH |
| Never `git add -a` | **HELD** — the stage list is §1 |
| Never commit or push from the agent | **HELD** — tree is modified-only; **Michael commits and dual-pushes** |

The instrument remains **decision-free**: `seen_keys` and `probes` still appear in no condition
under `_admission_lock` (`gate_g8f_the_fields_are_high_waters_and_decision_free` green at final
state), and the new update block sits **outside** that lock, on two plain locals, as before.

---

## 9. The complexity result — Beta-mandated phrasing, UNMODIFIED

> R-3's scaling model is gate- and benchmark-certified and strongly corroborated by
> Attempt 9's per-call cost, but its two dedicated production falsifier fields were not
> persisted and therefore were not observed in Attempt 9.

**No production observation is claimed by this commit.** After this repair lands, the first
production observation comes from the next naturally required Step-1 run — the Phase 7 soak.

---

## 10. Verification-integrity controls (VIR-1…6)

* **execution proof:** every tally in §7 comes from a run on VM101 whose log is retained under
  `/tmp/field6_logs/` (`base_*.log`, `final_*.log`); `rc` captured per suite; every suite prints
  its own completion sentinel. The M3 mutant carries an execution proof independent of its gate's
  verdict (§4).
* **clean control:** the 20-suite **baseline on the unmodified tree at `73633e7`**, taken before
  any edit — every red in §7 is attributed by differential against it, not by assertion.
* **fault-injection control:** three mutants, each proven APPLIED (anchor asserted), EXECUTED and
  DETECTED (§4); zero survived. Plus the `_bp_lock`-raises injection for edge case 4.
* **completion sentinel:** `52/52 checks green` / `COMPLETION SENTINEL: PASS` on the primary
  suite at final state.
* **unavailable-observer behavior:** this is the subject of the change — an unmeasured field is
  `None`/`UNOBSERVED`, **never `0`** (VIR-5: unobservable is not clean). Rows 2 and 4 of §6 are
  the two paths where that distinction is load-bearing.
* **audit claim scope:** the emitted `[S172-BP] summary` record and the two `_bp` fields behind
  it, on VM101's live tree. **No claim about production observation** (§9).
* **searched surfaces:** live `miner/range_miner_coordinator.py` (all four brief anchors
  re-derived by grep — they had **not** drifted from `d391a5c`); `_pump_deferred` in full;
  `tests/` (all 20 battery suites, plus a repo-wide hunt for transcribed tallies of the suites
  touched); `scripts/` for `S172-BP` extractors (**none**); `docs/` — the brief, the TB ruling it
  cites, `SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` §2/§3.1,
  `SESSION_CHANGELOG_20260814_S181.md` §5, and `/home/michael/dashboard_work/attempt6_logs/
  r2_battery.txt` for the battery list; `git show` for MP-1's and R-1's pinned anchors and for
  `git log 2c38f8c..HEAD -- miner/range_miner_coordinator.py`.
* **unavailable surfaces:** none required. **No GPU, no rig, no fleet, no launch** — this is a
  CPU-only observability commit and every gate is CPU-only.
* **governance trail searched:** `TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (the authority),
  `CLAUDE_CODE_INSTRUCTIONS_FIELD6_OBSERVABILITY_REPAIR.md`, the two session changelogs above,
  and the `tfm-project-facts` skill §2.19 / §2.24 / §2.33 / §7.
* **chapters searched:** none — no chapter-level claim is made.

---

## 11. Handover to Michael

1. **Stage exactly the six paths in §1.** Never `git add -a`; never build the list from recall.
2. **Read §5 before staging.** Three certified suites are edited outside the brief's enumerated
   scope. Two are mechanical; **§5.3 is a judgement call** and is the row to revert if Beta
   disagrees.
3. Gate 22 (§7) reds until those three files are committed. **Do not widen the allowlist.**
4. Then dual-push: `git push origin main && git push public main`.
5. **No production observation may be claimed until the Phase 7 soak** (§9).
