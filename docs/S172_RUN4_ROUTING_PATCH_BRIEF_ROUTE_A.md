# S172 — IMPLEMENTATION BRIEF: RUN-4 ROUTE-A ROUTING PATCH

**Status: REVISION 3 — R1 patch review returned DO-NOT-COMMIT on digest `0398c0d1…` with two
bounded executable blockers; both corrected. Operator authority now travels in a dedicated
channel populated only by the real CLI `--run-pipeline --params` seam (`split_operator_pin_params`,
MOVE not duplicate — justified at source), and capture separates key PRESENCE from usable VALUE,
failing loud on `None`/`''`/bool members. Adds G-ORIGIN, G-VALUE-USABLE and M4. Bundle is built in
sorted key order (Beta non-blocking item). Nothing committed.**

**Superseded status line: REVISION 2 — architecture APPROVED by Beta, four bounded edits incorporated. Nothing
committed, nothing launched.** For Michael and Team Beta. Written against
`TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md` (sha256 `ef3b701d6f51788b…`, 214 lines) and
`TB_RULING_RUN4_ROUTING_PATCH_BRIEF_REVIEW.md` (sha256 `362ba9e93a27cdb2…`, 183 lines), both read
from disk 2026-08-27. Base tree `69ca910`.

**Revision 2 incorporates the four required bounded edits:** (1) pipeline-invocation-local pin
ownership with explicit threading, scope expanded to `run_pipeline()` (§4.1, §1, gate 10b);
(2) `G-UNPINNED-IDENTICAL` compares against a **pre-edit clean control captured from untouched
`69ca910`** — the pinned-executable design is dropped (§6.1); (3) `G-EXACT` is value-and-type
equality, never object identity (§6 gate 2); (4) `M2` split into **M2a** (lifetime authority) and
**M2b** (default contamination) (§7). **STEP 0 is already executed** — the control artifact exists
and `agents/watcher_agent.py` was verifiably unmodified at capture time.

**Companions.** `S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` (the geometry and the four routes) ·
`S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md` (the volume bounds) · `LEADS.md` L-1, L-2
(both OPEN, both excluded by ruling §9).

**One falsifiable question:** *can the seven warm-start keys be made to travel the real WATCHER
path when — and only when — an operator supplies all seven explicitly, with no change whatsoever
to the command WATCHER builds when they are absent?*

---

## 0. AUTHORITY AND BINDING CONSTRAINTS

| source | what it binds here |
|---|---|
| ruling §1 | Route A **with containment**: routable only when explicitly operator-supplied; never defaults, resume state, WATCHER/LLM-generated, or carry-forward |
| ruling §2 | Run-4 values as manifest `default_params` **PROHIBITED**; declared ≠ defaulted; the invariant is byte/argument-equivalence when unpinned |
| ruling §3 | `_INTERNAL_ONLY_PARAMS` removal **only** on the explicit Step-1 human-supplied path; WATCHER manufacturing a value still fails closed; provenance marker required |
| ruling §5 | ten acceptance requirements; **all-seven-or-none BINDING**, partial pin REJECTED |
| ruling §8 | WATCHER-MANIFEST-ROUTING-1: repair **only** the seven warm-start declarations; **do not** wire the two threshold override arguments |
| ruling §9 | RAM ceiling and L-1 stay independent and OPEN; not folded in |
| skill §7 (owner rule) | when Beta offers multiple acceptable mechanisms, take the structurally **stronger** one — properties by construction, not by inference. Diff size is never a tiebreaker |
| CLAUDE.md 1, 3 | Claude does not commit and does not launch |

**Ruling §5 gate 3 note.** Gate 3 (`window_trials=1` yields the pinned trial, not a second TPE
sample) is an optimizer property, not a WATCHER-routing property. It is in scope for the
acceptance harness and out of scope for the code change. See §6.

---

## 1. SCOPE

**IN — three definitions in one file, plus module scope, a fixture and a test module.**

Scope rule (ruling §6, replacing the earlier "two definitions"): *the minimum definitions
necessary to capture the original operator bundle once, thread it invocation-locally, open
WALL 1 / WALL 2 only under that authority, and maintain declaration parity.* `run_pipeline()` is
in scope because the call graph requires it (§4.1) — **not** to enlarge the patch, and expressly
not hidden to keep the diff at two definitions.

```
agents/watcher_agent.py     run_pipeline()              (def 2417-...)   — capture + thread
agents/watcher_agent.py     run_step()                  (def 1443-1912)  — signature + two walls
agents/watcher_agent.py     _step1_declared_params()    (def 1290-1314)  — recognition parity
agents/watcher_agent.py     module scope                                 — one constant, one predicate
tests/fixtures/run4_routing_clean_control.py                             — DONE (STEP 0)
tests/fixtures/run4_clean_control_69ca910.txt                            — DONE (STEP 0 artifact)
tests/capture_run4_clean_control.py                                      — DONE (STEP 0 capture)
tests/test_s172_run4_routing_patch.py                                    — NEW, the acceptance gate
```

**The repo-root `watcher_agent.py` is a stale duplicate and is NOT touched.** It carries its own
`self.run_step(step, params)` at `:1547`. Nothing in this patch reaches it, and it must not be
"kept in sync" — the stale-duplicate question is separately ruled.

**OUT — explicitly, and each for a stated reason.**

| excluded | why |
|---|---|
| `agent_manifests/window_optimizer.json` `default_params` | ruling §2 PROHIBITED |
| `forward_threshold` / `reverse_threshold` args_map wiring | ruling §8; their own CLI help aborts the run (`WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED`) |
| `search_strategy`, `seed_count` args_map wiring | not authorized by any ruling; `search_strategy` is the §2.13 dead chain whose autonomous application is NOT approved, `seed_count` is the §2.29 parameter trap |
| `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py` | hops 2 and 3 are already complete and correct (§2). Touching them widens the surface for no gain |
| `distributed_config.json` `search_bounds` | Route B, REJECTED |
| L-1 float32/float64 seam · ingress byte bound | ruling §9 |
| the six-of-seven downstream completeness check | §3.4 — recorded as a finding, deliberately not repaired here |

**No GPU and no fleet are required to accept this patch.** Every one of the ten gates is provable
at or above the Optuna `enqueue_trial` seam, which precedes any sieve work. This matters: it means
the patch can close before Run 4 is scheduled, and it must not be bundled with a fleet run.

---

## 2. AS-BUILT ROUTING STATE — MEASURED 2026-08-27 AT `69ca910`

The chain is complete from hop 2 down and broken **only** at hop 1, in two independent places.

```
OPERATOR --params '{...}'
   |
   |  [WALL 1]  agents/watcher_agent.py:1548-1556
   |            final_params = {**default_params}
   |            allowed_params = set(default_params.keys())      <- 32 keys, ZERO warm_start
   |            if key in allowed_params: final_params[key] = value
   |            -> all seven DROPPED here, before any command exists
   v
   |  [WALL 2]  agents/watcher_agent.py:1840-1847
   |            _INTERNAL_ONLY_PARAMS = { ...8 names... }
   |            _cli_params = {k:v for k,v in final_params.items() if k not in _INTERNAL_ONLY_PARAMS}
   |            -> would strip them even if WALL 1 passed them
   v
 window_optimizer.py:1514-1528   --warm-start-* argparse, all seven, default=None   INTACT
   -> :1869-1875                 getattr(args, 'warm_start_*', None)                INTACT
     -> :695-701, :899-905        signature + forwarding, all seven                  INTACT
       -> window_optimizer_integration_final.py:2832-2848  _trial_history_ctx
          "[S166] explicit warm-start params -- override DB lookup"                  INTACT
         -> window_optimizer_bayesian.py:774-786  study.enqueue_trial(_ws_params)    INTACT
```

**Both walls must open together on the explicit path. Neither alone changes anything.** Opening
only WALL 2 leaves the keys dropped at 1548; opening only WALL 1 leaves them stripped at 1840.

Measured counts, reproducible:

```
manifest default_params                     32 keys, 0 warm_start
manifest args_map                           39 entries
args_map entries with NO default_params      11   <- the unroutable set
   of which warm_start_*                      7   <- authorized by ruling §1
   of which threshold overrides                2   <- forbidden by ruling §8
   of which other (search_strategy,            2   <- not authorized by anything
                   seed_count)
_INTERNAL_ONLY_PARAMS                         8 names
```

---

## 3. FOUR CORRECTIONS TO THE RULING'S IMPLEMENTATION SHAPE

Raised as evidence per the §7 working agreement. **None contests a disposition** — Route A, the
containment, the ten gates and the geometry all stand exactly as ruled. These concern only *where
the code goes*, and three of the four would produce a patch that passes review and does not work.

### 3.1 `_step1_declared_params` does not gate the command build

Ruling §2 states the preferred mechanism as *"`_step1_declared_params` should recognize the seven
already-declared manifest `args_map` entries as explicit allowable inputs."*

**That function does not sit on the execution path.** Measured by AST over the live file:

```
_step1_declared_params   def 1290-1314   called exactly ONCE, from line 1341
_ensure_execution_set    def 1316-1372   called from run_step line 1484
run_step                 def 1443-1912   WALL 1 at 1548-1556, WALL 2 at 1840-1847
```

`_ensure_execution_set` consumes `_step1_declared_params` only to answer *which backend and what
admission count* for the execution-set freeze. WALL 1 is a **separate, inline re-implementation**
of the same declared-key idea inside `run_step`, and it is the one the built command passes
through. A patch confined to `_step1_declared_params` would satisfy the ruling's literal wording,
review as correct, and route nothing.

**Resolution.** One canonical constant and one canonical predicate at module scope, consumed at
**all three** sites — WALL 1, WALL 2, and `_step1_declared_params`. The third is recognition
parity, so the two implementations of "declared" cannot drift; the first two are load-bearing.
This satisfies the ruling's intent and its wording, and it works.

### 3.2 `_INTERNAL_ONLY_PARAMS` holds eight names, not seven

```
warm_start_window · warm_start_offset · warm_start_skip_min · warm_start_skip_max
warm_start_fwd_thresh · warm_start_rev_thresh · warm_start_session_idx      <- the seven, routable
warm_start_session                                                          <- the eighth
```

`warm_start_session` has **no `args_map` entry** and appears nowhere in `window_optimizer.py`. It
has no CLI route at all. Routing it would emit `--warm-start-session` via the underscore→hyphen
fallback (`:1852`) — an argument the optimizer's parser does not define, i.e. an immediate
`SystemExit(2)` inside a launched production step.

**Resolution.** The patch narrows the strip to the exact seven. `warm_start_session` **remains
stripped unconditionally**, on every path, pinned or not. Gate 5b asserts this.

*(The run-4 proposal §2 says "strips all seven"; the live set is eight. Correcting the proposal's
count here so the patch is not written against it.)*

### 3.3 "args_map membership" is not a safe recognition rule

Ruling §2's phrasing — recognize *"the seven already-declared manifest `args_map` entries"* — is
exact and correct as a description of the seven. But implemented literally as a **rule** ("an
`args_map` entry with no `default_params` entry is an allowable explicit input") it admits all
**eleven** unroutable entries, including `forward_threshold` and `reverse_threshold` — the two
ruling §8 forbids in the same document — plus `search_strategy` and `seed_count`.

**Resolution.** The allowlist is an **exact enumerated frozenset of seven literal names** at
module scope, never derived from the manifest at runtime. A gate asserts the constant equals
exactly those seven and that the four hazardous names are absent from it (§6, gate 8b). Deriving
the set from `args_map` is the failure mode this brief exists to avoid, and it is the shape a
future refactor is most likely to reach for.

### 3.4 The downstream completeness check is six-of-seven — all-seven-or-none cannot be delegated

At `window_optimizer_bayesian.py:781`:

```python
_wsi = trial_history_context.get('warm_start_session_idx', 0)          # :780  DEFAULTS TO 0
if all(v is not None for v in [_ww,_wo,_wsk,_wsx,_wf,_wr]):            # :781  SIX values
    _ws_params = {..., 'session_idx': int(_wsi)}
    study.enqueue_trial(_ws_params)
```

`warm_start_session_idx` is excluded from the completeness check and **defaults to 0**. Run 4
requires `session_idx = 1` (midday); `session_idx = 0` is midday+evening.

**Consequence, and it is not cosmetic.** A six-key pin that drops only `session_idx` does not
fail — it enqueues silently with combined sessions, which the TB dataset-lifecycle ruling
(skill §2.10b) makes **non-certifying and prohibited by default**. The A/B would be destroyed and
the artifact would look pinned.

**A second, milder consequence — logging, not routing.** `window_optimizer_bayesian.py:785`
hardcodes `_ws_source = f'step1_trial_history (W{_ww}_O{_wo})'` and prints it at `:787`, so an
operator pin will be **misnamed in the optimizer's own logs as a trial-history warm-start**. The
enqueued values are unaffected. **WATCHER's `step1_pin_source` marker (§4) is the authoritative
provenance record**; the optimizer log line is not, and no acceptance evidence may be read off it.
Not repaired here — same scope boundary as the six-of-seven check.

**Resolution.** All-seven-or-none is enforced **at the WATCHER boundary**, fail-closed, before any
command is built — never delegated downstream. This is what ruling §5 gate 5 requires, and §3.4 is
the reason it cannot be satisfied by relying on the optimizer. The six-of-seven check itself is
**recorded as a finding and deliberately not repaired here** (out of scope, §1); it is a candidate
for WATCHER-MANIFEST-ROUTING-1's follow-up.

---

## 4. MECHANISM: DECLARED vs DEFAULTED — CHOSEN AND JUSTIFIED AT SOURCE

Ruling §2 offers two acceptable mechanisms. Under the owner rule the choice is not free.

| | **(a) sentinel/null in `default_params`** | **(b) recognition allowlist, no defaults** ← CHOSEN |
|---|---|---|
| how absence stays absence | the seven enter `final_params` as `None` at `:1548` (`{**default_params}`), then rely on the `if value is None or value == '': continue` skip at `:1862` to not emit | the seven never enter `final_params` at all when unsupplied |
| property holds by | **inference** over a downstream skip — which is why the ruling itself conditions it on *"only if tests prove absence remains absence"* | **construction** — there is no value to suppress |
| exposure to future edit | the keys sit in the manifest with values; the manifest layer has strong precedence, so a later editor filling them in makes a one-run pin silent production behavior — the exact hazard ruling §2 names | nothing declares a value anywhere; there is no field to fill in |
| gate-10 surface | must prove a null cannot become non-null across runs | vacuous — no persisted declaration exists |

**(b) is both Beta's stated preference and the structurally stronger option**, so the owner rule
and the ruling agree. Chosen.

**The mechanism, and why each property holds by construction:**

```
STEP1_EXPLICIT_PIN_KEYS : frozenset of the 7 literal names, module scope, never derived  (§3.3)

_step1_explicit_pin(step, operator_params) -> dict | None
    step != 1                      -> None            # step-scoped by construction
    none of the seven supplied     -> None            # gate 4: pre-patch behavior
    some but not all seven         -> RAISE, fail closed, naming supplied and missing   # gate 5
    all seven supplied             -> the bundle      # gate 1
```

### 4.1 Ownership and lifetime — BINDING (ruling Blocker 1)

> **The explicit-pin authority belongs to one pipeline invocation, not to the `WatcherAgent`
> object, the daemon, module state, `retry_params`, or persisted state.**

**Not `self._step1_pin_bundle`. Not a module global. Not daemon or persisted state.** WATCHER is
long-lived in daemon mode and the architecture deliberately separates daemon lifetime from
pipeline-invocation lifetime; instance state would silently give a later pipeline the earlier
one's pin authority.

**The call graph permits exactly one clean threading point.** Measured live:

```
run_pipeline(start_step, end_step, params)   def :2417   <- CAPTURE HERE, at entry
    :2474   results = self.run_step(step, params)        <- THREAD HERE, the ONLY caller
    :2525   retry_params = self._build_retry_params(_health, params)
    :2545   params = retry_params                        <- mutation happens AFTER capture
```

`run_step` has **exactly one caller in the module** (`:2474`), so threading is entirely internal.
External callers of `run_pipeline` — `chapter_13_triggers.py:616`, `:2993`, `:3426`, `:3467` —
**need no change**: capture happens inside `run_pipeline` from its own `params` argument.
`:2993` and `:3426` pass no `params` at all and therefore carry no pin authority, by construction.

**Lifetime, stated as four properties that hold by construction:**

1. **capture** — a defensive immutable copy of the seven operator values is taken at
   `run_pipeline` entry, **before** the retry/LLM loop can mutate `params`;
2. **own** — it lives in an invocation-local variable, nowhere else;
3. **thread** — it is passed explicitly to every Step-1 `run_step()` of that invocation, via one
   keyword-only parameter with a default, so no existing call signature changes;
4. **discard** — it dies when `run_pipeline` returns. A later pipeline on the same
   `WatcherAgent` begins with **no** pin authority unless its own operator input supplies all
   seven again.

**The discriminator for "explicitly operator-supplied" is that invocation-local frozen bundle,
not the mutable `params` dict.** This is the load-bearing design decision and it exists because
of a measured carry-forward path:

```
agents/watcher_agent.py:2068   retry_params = dict(original_params or {})    # carries operator params forward
agents/watcher_agent.py:2188   retry_params[_pname] = _pval                  # LLM proposals write here
agents/watcher_agent.py:2545   params = retry_params                         # rebinds the pipeline-level params
agents/watcher_agent.py:2474   results = self.run_step(step, params)         # next iteration sees the rebound dict
```

Reading `params` at `run_step` therefore **cannot** distinguish an operator pin from a
WATCHER-synthesized or LLM-proposed value. The operator's `--params` bundle is instead captured
**once, at CLI parse, into an immutable per-invocation record** — the same freeze-once posture
`freeze_execution_set` already takes toward the fleet — and the predicate consults that record.
Then:

- **gate 6** — a synthesized value is absent from the frozen record → fails closed. An operator
  who *did* supply the complete bundle replays legitimately, which is exactly what the ruling
  permits (*"unless the operator explicitly supplied the complete pin bundle"*).
- **gate 7** — LLM proposals write into `retry_params`, never into the frozen record → cannot
  create these keys. (Independently, `_build_retry_params` is Step-5-gated at `:2096-2100`, and
  step 5's manifest does not declare them; the frozen record makes it hold without depending on
  either.)
- **gate 10** — the record is per-invocation and never persisted. `daemon_state.json`
  (`:2714-2718`) is a separate surface and the patch writes nothing to it. **Gate 10's evidence
  must confirm that by reading the file, not by assuming it** (§6).

**Provenance marker (ruling §3), emitted only from the frozen-bundle path:**

```
step1_pin_source = explicit_operator_warm_start
```

on the `EXEC CMD` log record (`:1867`) and in the step's structured result. Absent — not `none`,
not empty — on every unpinned run, so its presence is positive evidence and its absence is the
pre-patch state. **It is a provenance record only: no decision logic consumes it** (the field-6
constraint, skill §2.52).

---

### 4.2 Operator-origin authority channel — BINDING (R1 Blocker 1)

`params` at `run_pipeline` entry proves a bundle EXISTED, not **who supplied it**. That is not a
theoretical gap: `chapter_13_triggers.py:616` is a live programmatic caller that passes `params`,
so authority inferred from `params` misclassified an agent-triggered run as an explicit operator
and stamped it `step1_pin_source=explicit_operator_warm_start`.

Authority therefore travels in its own channel:

- **`run_pipeline(..., *, _operator_pin_params=None)`** — keyword-only, defaulting to `None`, so
  every existing caller (`chapter_13_triggers.py:616`, the Chapter-13 dispatch, the selfplay path,
  the no-params CLI branches) acquires **zero authority with no change to its call site**.
- **`capture_step1_pin_bundle(operator_pin_params)`** consumes that channel and **never** ordinary
  `params`.
- **`split_operator_pin_params(params)`** is the ONLY seam that populates it, called from the real
  CLI `--run-pipeline --params` branch and nowhere else.
- **`assert_no_unauthorized_pin_keys(params, context)`** fails **loud** (`Step1PinAuthorityError`,
  `blocked_by=step1_unauthorized_warm_start_pin`) when the seven appear in ordinary `params`.

**The seam decision: MOVE, not duplicate.** Both paths were traced before choosing.

> **MOVE (chosen).** After the split the seven exist in exactly one place — the authority channel —
> on every path. That makes the fail-loud check **unconditional**: no authorized-invocation
> exemption, hence no branch in which the check is weakened, which is the ambiguity the ruling
> exists to remove. It also settles the retry question: `_build_retry_params` copies
> `original_params` wholesale (`:2234`), so with the seven absent from that dict the retry path
> **cannot re-carry them**, and the frozen bundle stays the single source — which is what
> "legitimate retry keeps the frozen bundle" has to mean.
>
> **DUPLICATE (rejected).** The seven would remain in ordinary `params` on a legitimate pinned
> invocation, so fail-loud would trip **the authorized run itself**. Avoiding that needs an
> exemption branch keyed on authority — reintroducing exactly the "authority inferred from params"
> coupling Blocker 1 removes — and `retry_params` would re-carry the seven at `:2234`, restoring
> the second source the frozen bundle was created to eliminate.

**`G-UNPINNED-IDENTICAL` is unaffected**: with no warm-start input the split pops nothing,
`_operator_pin_params` is `None`, and ordinary `params` is unchanged. Verified — still list-equal
at 47 tokens against the pre-edit `69ca910` capture.

### 4.3 Presence vs usable value (R1 Blocker 2), and one deliberate superset

`present = STEP1_EXPLICIT_PIN_KEYS ∩ params.keys()` separates **key presence** from **usable
value**. Empty → no pin. Otherwise all seven must be present **and** none may carry a value the
step-1 command builder treats as absent — any violation fails loud before a command is built. An
explicit `None` or `''` member is a **malformed pin that fails**, not a key that quietly vanishes
and collapses the request to "unpinned".

`_step1_pin_value_defect(value)` derives the rejected set from the builder's OWN semantics at
`agents/watcher_agent.py:2009-2020` — which values never reach the routing exit at `:2020`:

| value | builder line | outcome |
|---|---|---|
| `None` | `:2017` | skipped — never reaches argv |
| `''` | `:2017` | skipped — never reaches argv |
| `False` | `:2013` | bool branch omits the flag entirely |
| `True` | `:2011` | emits a **valueless** flag |

**FLAGGED FOR REVIEW — this is a superset of the ruling, not a substitution.** Beta's Blocker 2
names `None` and `''`. The two **bool** cases are rejected as well, because they are the identical
defect class — present but non-routable — reached through the builder's other branch, and leaving
them would ship a known hole in the exact property the blocker is about. For `True` the flag is
emitted with no value while all seven are declared `type=int`/`type=float` at
`window_optimizer.py:1514-1526`, so argparse either consumes the next token as this flag's value or
aborts the dispatched step — which of the two depends on what follows in argv; the command is
corrupted either way. Values the builder stringifies normally are **out of scope**: this check is
builder-skip semantics, not general value sanity. If Beta prefers the literal two, it is a two-line
change in `_step1_pin_value_defect`.

---

## 5. IMPLEMENTATION SPECIFICATION

Four definitions changed and eleven added at module scope, one file, plus the CLI seam. No new
`def` inside `run_step`. The scope rule is the ruling's: *minimum definitions necessary* — R1
Blocker 1 required the operator-origin channel, so `run_pipeline`'s signature and the
`--run-pipeline` CLI branch are in scope.

1. **Module scope** — `STEP1_EXPLICIT_PIN_KEYS` (frozenset, 7 literals),
   `STEP1_PIN_SOURCE_MARKER`, the two provenance key names
   (`STEP1_PIN_PROVENANCE_KEY` = `step1_pin_source`, `STEP1_PIN_ARGV_KEY` =
   `step1_pin_argv`, named once so the gate binds to the production names),
   `Step1PinBundleError`, **`Step1PinAuthorityError`**,
   **`_step1_pin_value_defect(value)`** (§4.3),
   `capture_step1_pin_bundle(operator_pin_params)` — **consumes the authority channel, never
   ordinary `params`** (§4.2) — **`split_operator_pin_params(params)`**,
   **`assert_no_unauthorized_pin_keys(params, context)`**,
   `_step1_explicit_pin(step, pin_bundle)` and
   `_step1_stamp_pin_provenance(results, pin, argv)` per §4. The bundle is built over
   `sorted(STEP1_EXPLICIT_PIN_KEYS)` so `step1_pin_argv` is stable across runs.
2. **`run_pipeline`** — gains keyword-only `_operator_pin_params=None`, the operator-authority
   channel (§4.2). At entry it (a) calls `assert_no_unauthorized_pin_keys(params, …)` — fail-loud
   if the seven appear in ordinary `params`; (b) captures the defensive immutable bundle **from the
   authority channel**, before the retry loop; (c) holds it invocation-locally and threads it to
   each Step-1 `run_step` via one keyword-only parameter with a default. Discarded on return
   (§4.1). **No instance attribute is created**, and a gate asserts that.
2b. **CLI `--run-pipeline` branch** — the ONLY seam that populates the authority channel. Calls
   `split_operator_pin_params(override_params)` to **MOVE** the seven out of ordinary params, then
   passes them as `_operator_pin_params`. No other call site in the codebase may do this.
3. **WALL 1 (`:1548-1556`)** — when the predicate returns a bundle, the seven join `final_params`
   **from the bundle**, not from `default_params`. When it returns `None`, the merge is untouched.
4. **WALL 2 (`:1840-1847`)** — `_INTERNAL_ONLY_PARAMS` narrows to the seven **only** when the
   predicate returned a bundle. `warm_start_session` is stripped unconditionally in both cases
   (§3.2). Unpinned, the eight-name strip is byte-identical to pre-patch.
5. **`_step1_declared_params` (`:1290-1314`)** — same recognition, for parity (§3.1). It is not on
   the command path; the change exists so the two "declared" notions cannot drift.
6. **Provenance** — on the pinned path only, and in **both** surfaces §4 promises: the `EXEC CMD`
   log record, **and the step's structured result**, where `_step1_stamp_pin_provenance` inserts
   `step1_pin_source` (authority) and `step1_pin_argv` (what that authority requested) at every
   dispatch-outcome return of `run_step` — success-with-results, basic success, non-zero return
   code, timeout, and execution error. Both keys are **ABSENT**, never `None`, on the unpinned
   path, so `'step1_pin_source' in results` is the whole test. Nothing is written back to any
   results **file** and nothing reaches `daemon_state.json`; the stamp is on the in-memory dict
   returned to `run_pipeline`, which consumes it only via `.get()` and `evaluate_results`.

**The invariant every one of these must preserve, stated as ruling §2 states it:**

> No explicit warm-start input → execution command is byte/argument-equivalent to the current S167
> behavior for these seven fields.

---

## 6. ACCEPTANCE CRITERIA — THE TEN §5 GATES

`tests/test_s172_run4_routing_patch.py`. Every gate terminates `PASS | FAIL | UNAVAILABLE |
INCOMPLETE`; only `PASS` accepts (VIR-3). No gate asserts a tally; each asserts a behavior.

| # | ruling §5 requirement | gate | how it is proven |
|---|---|---|---|
| 1 | seven pins survive the full chain | **G-CHAIN** | drive `--params` with all seven → assert all seven `--warm-start-*` present in the built `cmd`; then drive the optimizer to study creation and assert the enqueued dict carries all seven. Stops above any sieve work |
| 2 | each enqueued value equals the requested value exactly | **G-EXACT** | **value AND type equality** — `type(got) is type(want) and got == want` — explicitly **never** Python object identity (`is`), which for `0.71` or a boxed int is an implementation accident, not a semantic check. Assert `0.71`/`0.47` survive the `float()`/`round(…,2)` seam unchanged; assert `session_idx == 1` **as an `int`**, not the `:780` default of 0; assert no int→float or float→int coercion occurred |
| 3 | `window_trials=1` yields the pinned trial, not a second TPE sample | **G-ONE-TRIAL** | assert exactly one trial is created and that its params are the enqueued bundle. *Optimizer property; harness-only, no code change* |
| 4 | omitting all seven → pre-patch command behavior | **G-UNPINNED-IDENTICAL** | build `cmd` on the patched tree with no warm-start keys, using the **identical fixture**, and compare **list-equal** against the **STEP-0 pre-edit capture** (§6.1). The pinned-executable design is **dropped entirely** |
| 5 | partial pin rejected | **G-PARTIAL-CLOSED** | all 126 non-empty proper subsets of the seven → each must fail closed naming supplied and missing. Includes the six-key subset that drops `session_idx`, which §3.4 shows would otherwise enqueue combined sessions silently |
| 5b | *(derived, §3.2)* the eighth key never routes | **G-EIGHTH** | `warm_start_session` absent from `cmd` on both the pinned and unpinned paths |
| 6 | retry/resume cannot synthesize or replay pins | **G-NO-SYNTH** | inject the seven into `retry_params` at the `:2068`/`:2545` carry-forward with an **empty** frozen record → assert fail-closed and no `--warm-start-*` in `cmd`. Then repeat with a **complete** operator bundle in the record → assert the legitimate replay is permitted |
| 7 | WATCHER/LLM cannot create these keys | **G-NO-LLM** | drive the `:2188` proposal path with each of the seven as a proposed parameter → assert none reaches `cmd`, on both a step-1 and a step-5 dispatch |
| 8 | execution record identifies them as explicit operator pins | **G-PROVENANCE** | Two surfaces, both directions. **Log arm:** `step1_pin_source = explicit_operator_warm_start` parsed off the emitted log record, present pinned and **absent** (not empty, not `none`) unpinned. **Structured-result arm (§4):** the dict `run_step` actually returns carries `step1_pin_source` equal to the marker and `step1_pin_argv` **list-equal to the argv actually built**, and on the unpinned dispatch carries **neither key** — `in` is the test, so a `None` placeholder fails it. The argv arm deliberately does **not** assert warm-start flags are present: authority and routing are separate facts and routing is gate 1's. That is what keeps this gate green under **M1b**, where the pin is genuinely authorized and genuinely stripped, and it is why a green marker can never be read as proof of routing |
| 8b | *(derived, §3.3)* the allowlist is exact | **G-ALLOWLIST-EXACT** | `STEP1_EXPLICIT_PIN_KEYS` equals exactly the seven literals; `forward_threshold`, `reverse_threshold`, `search_strategy`, `seed_count` are each absent; AST arm asserts the constant is a literal frozenset and is **not** constructed from manifest data at runtime |
| 11 | *(R1 Blocker 1)* pin authority is ORIGIN, not presence | **G-ORIGIN** | three arms. (1) the complete seven in ordinary `run_pipeline(params=...)` with no authority → bundle is `None`, **0/7** routed, run BLOCKED with `blocked_by=step1_unauthorized_warm_start_pin`, no provenance key on the returned dict; (2) the same seven through the authority channel → **7/7** + provenance, and the CLI seam is asserted to have **MOVED** them (zero of the seven left in ordinary params — a duplicate fails the gate); (3) `inspect.signature` asserts `_operator_pin_params` is **keyword-only with default `None`**, so the live programmatic callers (`chapter_13_triggers.py:616`, the Chapter-13 dispatch, the selfplay path) acquire zero authority without any change to their call sites |
| 12 | *(R1 Blocker 2)* present-but-non-routable values fail loud | **G-VALUE-USABLE** | sibling to gate 5, which tests ABSENT keys. **30 negative cases** — each of the seven individually `None`, `''`, `False`, `True`, plus all-seven-`None` and all-seven-`''` — each must raise, and a **silent collapse to "unpinned" also fails**: a malformed explicit pin must fail, not vanish. Non-vacuity: the same shape with usable values must be ACCEPTED, in **sorted key order** |
| 8c | *(second-reader lock, §3.1)* the declaration-parity mirror is inert for fleet resolution | **G-PARITY-INERT** | `_step1_declared_params` mirrors the seven on **any** params dict carrying them, with or without pin authority, so its inertness is load-bearing and is locked by gate rather than by inspection. Three arms: (1) the mirror is live — all seven arrive in `declared` with values intact, and none is invented when the caller supplied none; (2) the kwargs actually handed to `resolve_execution_set` by the **real** `_ensure_execution_set` are identical with and without the seven; (3) this gate's own fault-injection control — flipping `use_range_miner`/`use_persistent_workers` moves `backend` `miner`→`pwc`, proving arm 2's equality is not vacuous. Nothing contacts the fleet |
| 9 | a mutation restoring unconditional stripping is detected | **M1 + M1b** | §7 |
| 10 | a mutation persisting a pin as a default into a later unpinned run is detected | **M2a + M2b** | §7 |
| 10b | *(ruling Blocker 1)* invocation isolation | **G-INVOCATION-ISOLATION** | one `WatcherAgent` instance → `run_pipeline` with all seven completes → **second `run_pipeline` with no warm-start keys** → assert zero `--warm-start-*` in the second argv **and** zero pin provenance. Two `run_pipeline` invocations, not two `run_step` calls — the weaker shape cannot see instance-state leakage |

**Gate 10 additionally requires a positive read, not an assumption:** after the pinned invocation
followed by the unpinned one on the same instance, assert `daemon_state.json` (`:2714-2718`)
contains none of the seven, **and** assert no attribute on the `WatcherAgent` instance holds them.
An unreadable state file reports `UNAVAILABLE`, never `PASS` (VIR-5).

### 6.1 The G-UNPINNED-IDENTICAL oracle — captured, not reconstructed

Ruling Blocker 2. The original design executed `run_step` source pinned at `69ca910` against the
live namespace. That violates **EXEC-PIN-1**, and concretely: pinned `run_step` calls
`_ensure_execution_set` → `_step1_declared_params`, **which this patch changes**. The gate meant
to prove "unpinned is unchanged" would consume the post-patch helper and could false-green.

**Executed instead, before the first edit, per Beta's strong preference — and it is DONE:**

```
artifact          tests/fixtures/run4_clean_control_69ca910.txt
base commit       69ca9100f72adbeaddceddae1f11c09909b8e0c3
target digest     8d1ef6d0b7bf1f093a1fbb6bb5562b538ef91074efef3a07a7b5706ff1f6d716
                  (worktree == 69ca910 byte-for-byte; git status empty at capture)
manifest digest   cb901d6bb64af2a1d2b81475dd8902bf58ede866a35c6d889a3d41770182b3d7
fixture module    tests/fixtures/run4_routing_clean_control.py
                  sha256 8f61420c8b36096a0537f03346aeacfdff48912a86b175af67416481d6fb91ce
argv_unpinned     47 tokens, recorded verbatim
payload sha256    b1ca456c2beb95350fe6fc1620d854692c9c062f29705e75d955aeebdbbc0f81
sentinel          CLEAN_CONTROL_CAPTURE_COMPLETE
```

**No historical code is ever executed.** The gate reads a recorded list and compares.

**The fixture module is the contract.** Capture and gate import the same module; the gate
re-verifies its sha256 against the artifact and reports `INCOMPLETE` on mismatch, so the oracle
cannot drift by an edit to the harness.

**The stub boundary is part of the fixture and is recorded in the artifact** — every stubbed
collaborator (`_ensure_execution_set`, `_run_preflight_check`, `check_output_freshness`, the two
P0.5 helpers, six `miner.dataset_authority` functions, `database_system.DistributedPRNGDatabase`,
and the `_run_step_streaming` interception point) is one this patch does **not** change, stubbed
because it reaches live host state that is not deterministic across the capture/gate interval —
notably the S145 certified cursor, **which Run 4 itself would advance**.
`assert_seed_domain_preflight` is deliberately **not** stubbed: pure arithmetic, so the real one
runs. An early guard firing raises `INCOMPLETE`, never an empty argv (VIR-5).

**A second, unplanned result the capture produced — executable proof of the dead chain.** The
capture also built the argv with all seven values *supplied*, pre-patch. It is **byte-identical to
the unpinned argv, 47 tokens, zero `--warm-start-*` tokens.** Brief §2's dead-chain claim is
therefore no longer only a source reading; it is a recorded measurement, and it is retained in the
artifact as `argv_supplied_seven_DIAGNOSTIC`. **It is diagnostic only and is never a gate-4
oracle** — post-patch that same input legitimately produces a different argv, which is the point
of the patch.

**Non-vacuity is asserted, not assumed.** Each gate asserts its fixture actually exercised the
condition — G-CHAIN asserts seven non-`None` values were supplied; G-PARTIAL-CLOSED asserts 126
subsets ran; G-UNPINNED-IDENTICAL asserts the recorded control artifact was read **and** that the
fixture module's live sha256 still equals the one recorded at capture — no historical source
is located or parsed anywhere, that design having been dropped (§6.1). A gate that cannot
establish its own precondition reports `INCOMPLETE`.

---

## 7. MUTATION EVIDENCE

Project standard (§2.52 `G-MUT-FIELD6`): every mutant **APPLIED, EXECUTED, DETECTED**, each
rebound against **production module globals** so it cannot escape into the test module's namespace
(the A8-B2 escape). `_run_gate_under_mutant` credits only the gate's own terminal verdict — a
gate that *raises* under a mutant returns `RAISED`, which is neither detection nor
still-green, and the mutant terminates `INCOMPLETE`. *Any* exception counting as detection is
the recorded vacuity failure (§2.44).

| mutant | shape | must red | must NOT red |
|---|---|---|---|
| **M1** | restore WALL 2 to the unconditional eight-name strip | G-CHAIN, G-EXACT | G-UNPINNED-IDENTICAL (unpinned behavior is unchanged by M1 — this asymmetry is what proves the two gates are independent) |
| **M1b** | *(second-reader lock)* the **surgical** WALL-2 defect: the narrowing at WALL 2 is killed **in source**, with `STEP1_EXPLICIT_PIN_KEYS` and `capture_step1_pin_bundle` untouched | G-CHAIN, G-EXACT | G-UNPINNED-IDENTICAL, **G-ALLOWLIST-EXACT**, **G-PARTIAL-CLOSED**, **G-PROVENANCE** |
| **M2a** | *lifetime authority*: the invocation-local bundle becomes process/agent-lifetime authority (e.g. `self._step1_pin_bundle`), so a second pipeline on the same instance inherits it | G-INVOCATION-ISOLATION, gate-10 instance-attribute read | G-CHAIN, G-UNPINNED-IDENTICAL *within a single invocation* — M2a is invisible to any single-invocation gate, which is exactly why gate 10b is two `run_pipeline` calls |
| **M2b** | *default contamination*: the pins are written into ordinary/default `final_params` and reappear on a subsequent unpinned invocation | G-UNPINNED-IDENTICAL, G-PROVENANCE, gate-10 state read | G-CHAIN |
| **M3** | derive `STEP1_EXPLICIT_PIN_KEYS` from `args_map` orphans instead of literals (§3.3) | G-ALLOWLIST-EXACT | G-CHAIN, G-EXACT (all seven still route — the defect is the four *extra* names) |
| **M4** | *(R1 Blocker 1)* revert the capture source to ordinary `params` — the exact pre-R1 defect. The fail-loud guard is disabled **alongside** it, because in the pre-R1 shape it did not exist; reverting only one half would be a strawman no real regression could produce | G-ORIGIN | G-CHAIN, G-UNPINNED-IDENTICAL, G-VALUE-USABLE |

**The structured-result arm was verified independently of the log arm.** `G-PROVENANCE`
short-circuits on the log arm, so under **M2b** the structured arm would otherwise never execute
and its detection power would be untested — the precise vacuity shape §2.44 names. Probed directly
with the log arm bypassed: on a clean tree the unpinned structured result carries neither key; under
M2b it carries **both**, with `step1_pin_source=explicit_operator_warm_start` and a **61-token**
contaminated argv on a dispatch that supplied no warm-start keys at all; restored, neither key
returns. The arm reds on its own.

**M1 and M1b are both required, and M1b is the load-bearing one.** M1 empties the allowlist constant, which has *two* consumers — WALL 2 **and** `capture_step1_pin_bundle` — so its blast radius is wider than its label and it cannot, alone, prove WALL 2 is the thing being tested. M1b kills WALL 2's narrowing **in source** and nothing else, so `G-ALLOWLIST-EXACT` and `G-PARTIAL-CLOSED` stay green; that green is the evidence that the reddened gates reddened for the WALL-2 reason. **`G-PROVENANCE` also stays green under M1b**, and that is the sharpest result in the suite: WALL 1 still fires, so the pin is accepted and its provenance logged **while nothing routes**. The marker therefore proves authority and never routing — exactly the division ruling §5 assigns it, now proven by mutant rather than asserted. The harness mutates production source, preserves line count so every `file:line` anchor stays valid, restores the original bytes in `finally`, and **verifies the restore by sha256** — a mutation harness that cannot prove restoration has left the tree in an unknown state.

**M2a and M2b are separate defects and must not be merged back into one mutant.** M2a is a
*lifetime* failure and M2b a *storage* failure; each is invisible to the other's gate. Under both,
**the pinned path stays green** — Run 4 would work while S167 containment was silently destroyed,
which is the dangerous shape ruling §4 names.

**M3 is not optional.** It is the mutant for the failure this brief was written to prevent, and it
is the one a reviewer reading only the pinned-path gates would never catch: under M3 every gate
about the seven stays green while the two threshold overrides ruling §8 forbids become routable.

---

## 8. WHAT MUST NOT BE TOUCHED

- **`default_params`** — ruling §2. No sentinel, no null, no key.
- **The two threshold override arguments** — ruling §8. They stay inert. Their CLI help aborts the
  run by design; making them reachable would arm that abort.
- **`search_strategy`, `seed_count`** — no ruling authorizes them; both have their own hazard
  history (skill §2.13, §2.29).
- **Hops 2 and 3** — complete and correct at `69ca910` (§2). No edit.
- **The six-of-seven downstream check** — recorded (§3.4), not repaired.
- **L-1 and the ingress byte bound** — ruling §9.
- **`_repository_state` / D3.5 clean-tree semantics** — untouched, as always (§2.35).

---

## 9. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

```
execution proof:            every gate prints its own PASS/FAIL line and the suite emits a
                            completion sentinel; a suite that dies mid-run cannot read as pass
clean control:              unpinned run on the patched tree -> cmd list-equal to 69ca910-pinned cmd
fault-injection control:    M1, M2a, M2b, M3 -- each APPLIED, EXECUTED, DETECTED, rebound against
                            production module globals
completion sentinel:        pass-only sentinel line at suite end
unavailable-observer:       unreadable manifest, unlocatable pre-patch source, or unreadable
                            daemon_state.json -> UNAVAILABLE / INCOMPLETE, never PASS
audit claim scope:          WATCHER hop-1 routing for exactly seven named keys at 69ca910.
                            NOT a claim about optimizer behavior below enqueue_trial, and NOT a
                            claim that Run 4 will pass
searched surfaces:          agents/watcher_agent.py (live, AST + read), window_optimizer.py,
                            window_optimizer_bayesian.py, window_optimizer_integration_final.py,
                            agent_manifests/window_optimizer.json (parsed, not grepped -- the
                            shell grep wrapper ignores *.json), git status, docs/ governance
                            trail, docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md
unavailable surfaces:       ser8 pre-repository archive (no credential from VM101); the rigs
                            (not required -- this patch touches no deployed worker file)
governance trail searched:  TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md (full, verbatim),
                            TB_RULING_BRIEF_I_*, TB_RULING_WINDOW_ANCHOR_*, run-4 proposal,
                            LEADS.md
chapters searched:          not applicable -- no chapter documents WATCHER param routing;
                            stated rather than left silent (VIR-6 addendum)
```

---

## 10. GOVERNANCE

- **SR-1 does not bind.** The patch changes no definition in `miner/range_miner_coordinator.py`,
  so no historical `DECLARED_CHANGED` set is affected. Stated explicitly rather than left silent,
  because the check is the operational step (skill §7).
- **SR-2.** Changelog is `SESSION_CHANGELOG_20260827_RUN4_ROUTING_PATCH.md`. No S-number.
- **Gate 22 will red while the new test file is untracked.** Expected, not a regression, and not a
  reason to widen the allowlist. It self-clears on commit. Same answer as every prior occurrence.
- **Clean tree.** Seven working-tree entries exist today (four docs from the prior session, this
  brief, the ruling, and the patch's own files). All are Michael's to commit. Clean-tree admission
  reads `git status --porcelain`, so **modified-tracked trips it too** — this is what refused
  attempt 3.
- **Environment note, not patch scope.** `tests/test_watcher_llm_integration.py` could not run
  during the first verification pass (`ModuleNotFoundError: pytest` in `~/venvs/torch`). Michael
  installed pytest (9.1.1); the suite is **13/13 passed against the patched tree**, with 4
  `PydanticDeprecatedSince20` warnings from class-based `config` in `agents/contexts/`
  `base_agent_context.py:43`, `window_optimizer_context.py:27` and `agents/full_agent_context.py:26`.
  Those three files are **untouched by this patch** and the warnings pre-date it; they are recorded
  here as an environment observation and are **not** chargeable to the differential. The venv
  dependency change belongs in the committed env-capture artifact (box contract §4).
- **Nothing is committed by Claude. Nothing is launched by Claude.**

---

## 11. DELIVERABLES AND REVIEW

1. This brief → Beta.
2. On approval: the patch (`agents/watcher_agent.py`) + the suite
   (`tests/test_s172_run4_routing_patch.py`), with the full gate output and the four mutant
   results (M1, **M1b**, M2a, M2b, M3), plus the differential-worktree proof against `69ca910` so only the differential is
   chargeable to the change.
3. Beta review of the patch. **Then** Michael commits and dual-pushes.
4. Run 4 is a separate decision, after the routing change **and** infrastructure instrumentation
   are both verified (ruling FINAL DISPOSITION).

**Target: ≤3 review rounds** (skill §9.22). The three most likely defect shapes have been
pre-identified and are each pinned by a gate or a mutant: patching the non-executing function
(§3.1 / G-CHAIN), opening the eighth key (§3.2 / G-EIGHTH), and deriving the allowlist from
`args_map` (§3.3 / M3).

**The claim boundary this patch inherits and must not exceed:** it makes the pin *routable*. It
does not make Run 4 correct, does not certify population equivalence, and resolves neither L-1 nor
the ingress byte bound. Ruling §7's mandated phrasing governs every later statement about the A/B.
