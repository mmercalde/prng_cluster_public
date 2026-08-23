# SESSION CHANGELOG — 2026-08-22 — BRIEF I PRODUCTION-SHAPE PROOF AND BOUNDED REPAIR

*(SR-2: date + topic, no S-number. Second cycle of 2026-08-22; the first is
`SESSION_CHANGELOG_20260822_WINDOW_ANCHOR_BRIEF_I.md`, committed at `48a8705`.)*

**Base commit:** `48a8705` — unchanged throughout. Nothing committed by Alpha.

## What happened

Beta's post-commit closure steps 1–3 were executed against `48a8705`. Steps 2a/2b passed; the
production-shape run **failed**, exposing a real Brief-I defect; Beta ruled; a bounded repair was
implemented, reviewed and approved for commit.

## Closure steps 2a / 2b — PASS

- **Fleet parity 30/30.** Pre-deploy the three rigs were uniformly at **`e9ca800`** — they had
  never received the field-6 repair `d8b21e3` either. Deployed all ten governed files at
  `48a8705`, verified on target, each host naming itself.
- **Host/worker schema parity 3/3, 0 failed checks.** A behavioural probe (not introspection):
  the deployed worker actively **rejects** a legacy `offset` key and **refuses** a missing
  `generator_phase` rather than defaulting it.
- The parity gate's AST cross-check also proved **Brief I added no uncovered project import** to
  the worker's closure.

## The production-shape run — `distributed_config_t1_eed23c7f`, FAILED

All six pre-launch gates green. Then:

```
128/128 stripes over four phases · trial committed · 0 lease expiries · 0 disconnects
serve loop 713.663 s · iteration_max 0.796 s  (MP-1's pathology was 940.971 s)
staging 5632 jobs @ 7.892/s · 0 pauses · 0 capacity terminations
window_anchor_val=58 / generator_phase=0 across coordinator -> 25 workers -> return path
```

**Both field-6 falsifiers were observed in production for the first time — 30 and 126, neither
`UNOBSERVED`.** Beta records that observation as now legitimately made; the §2.49 mandated
phrasing about R-3 no longer waits on it.

**Terminal defect:** `KeyError: 'offset'` at `utils/canonical_records.py:217` in
`build_mode_records`, via `commit_trial → assemble_trial → merge_validated_spools`. Option-C
retention, the fail-closed assembly and the ingress wall all behaved as certified — only the first
fault was a defect, and that retention is what made offline diagnosis possible.

**A hypothesis Alpha raised, tested and refuted** is kept in the record: `_CONTEXT_FIELDS` at
`:1038` was **not** involved — the manifests carried `window_anchor=58`, `generator_phase=0`, no
`offset`, and 5632/5632 published cleanly.

## The bounded repair (authorized, reviewed, approved)

**Two production sites, not one.** Alpha's first diagnosis named `:217` only; the regression sweep
found `normalize_trial_populations` at `:369` — the PWC/ZMQ wrapper — building its own context with
the retired key. **Alpha initially misreported the resulting red as stale fixtures; it was a second
unmigrated production consumer.** Beta ratified it as in-scope and stated the invariant:

> Every post-F-4 producer of canonical record field `"offset"` on these migrated paths sources it
> from `window_anchor`, never from a retired context `offset` and never from `generator_phase`.

**Beta's semantic ruling, encoded in code and docs — Alpha's rationale was overruled:** array 4
`offset` is a **legacy wire name** whose one post-F-4 meaning **is** the window anchor, **at any
phase value** — not merely while v1 pins phase to 0. Alpha's "coherent only because phase == 0"
was removed from every artifact, with a visible correction trail.

**Frozen contract untouched:** 22 arrays, index 4 still `"offset"`, no `window_anchor` array, no
23rd array, phase never fed into array 4.

**New gate `G-PHASE5-ASSEMBLY`** — distinct from `G-PHASE5-SEAM`, which is unchanged and unwidened
(AST body sha `f5781a682299a82e`, identical to HEAD). Reaches the real
`assemble_trial`/`build_mode_records` surface. Fixture values deliberately unequal —
`window_anchor=58`, `generator_phase=0` — asserting `"offset" == 58`, never `0`. Two-directional
non-vacuity: restoring `ctx["offset"]` and sourcing from `ctx["generator_phase"]` are **both
DETECTED**.

**Two certified-suite fixture migrations, ratified under the field-6 precedent**, with
**zero assertion additions/removals/changes proven by assert-line diff** (d3_25 114/114,
d1_engine 186/186). Disclosed limitation: both fixtures keep anchor and legacy value equal, so
neither discriminates anchor from phase — `G-PHASE5-ASSEMBLY` carries that.

## Batteries

```
brief_i 26/26 · brief_i_mutants PASS · d3_25 13/13 (== baseline) · columnizer 10/10
finalizer 60/60 · d6_2 31/31 · chapter2 12/12 · phase3_worker 18/18
```

**`test_s172_phase5_d1_engine`, in Beta's binding wording (§5):**

> **same pre-existing red population / changed failure depth / no demonstrated new production
> regression**

18 failures before the repair and 18 after, measured in a `git worktree` at `48a8705` in the same
environment. Some failures moved from the mandatory-context guard into the already-known RC-1
path, so the depth changed even though the count did not. **No RC-1 repair authorized.**

## Findings filed as separate governed items

| item | state |
|---|---|
| `OBSERVABILITY_GAP_1` | `commit_trial:8681` captures the exception to an in-memory dict; zero Phase-5 log lines; the ledger persists only three-state status. Third instance of the F2 class. Filed, **not repaired here** |
| `B7` | Same mechanism, `0/5632` matching `0/29,082`. **Acceptance classification DEFERRED** until a run publishes |
| `HARNESS-LEDGER-ORDER-1` | The harness cannot bootstrap a fresh ledger — the sampler requires the ledger the coordinator creates. Filed |
| `DEP-ABI-V2-NPZ-SEMANTICS` | Recorded in `BACKLOG.md` as an **audit dependency**; no 22-array amendment pre-authorized |
| **nested-tally leak** | `ac7_final/SUMMARY.tsv`'s `d1_engine` row carries a neighbouring suite's tally verbatim. **Not admissible baseline evidence.** Beta's standing rule: a summary-extracted tally is not authoritative unless bound to that suite's own completion sentinel |

## Two operational deviations — both RATIFIED

- **Ledger displacement** (PROD-SCOPE-1 exception): hash → copy outside staging → re-hash →
  functional read → only then move originals aside, WAL included. Nothing deleted; two
  byte-identical copies. A one-time governed migration, **not precedent**.
- **Ledger pre-creation** through the production `MinerLedger` constructor, with birth provenance
  captured before launch: sha `74c0b150…`, **0 rows**, post-separation schema. Every row in the
  run's ledger is therefore attributable to the run.

## Retention

Preserved and untouched throughout, including across a read-only offline reproduction:
**6151 staging files · C2's 512 orphans · the archived 9-run ledger · ledger sha unchanged.**

## fallback parity

`fallback parity: code=[UNKNOWN — not measured this session], env=[UNKNOWN]` — pass 2 needs `.127`
booted and Zeus runs one OS at a time; unchanged by this work.
