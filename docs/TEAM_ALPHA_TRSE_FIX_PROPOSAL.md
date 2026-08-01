# TEAM ALPHA → TEAM BETA — TRSE Step 0: audit result and proposed fix

**Re:** `docs/TRSE_STEP0_AUDIT_v1.md`. Audit only — no code, config or documentation was
changed. **This document proposes a fix; nothing has been implemented.** Requesting Beta's
review before any repair pass.

---

## 1. The headline: TRSE's mathematics is sound

The audit's own summary:

> TRSE's **analytical core is sound and its one applied rule is correctly wired end-to-end.**

**Rule A is verified working**, producer → consumer, this session: `regime_type`,
`regime_type_confidence` and `regime_stable` are produced at `trse_step0.py:397`/`:740`, read
at `window_optimizer_bayesian.py:497-499`, evaluated in the live gate at `:508-510`;
`bounds.max_window_size` is mutated at `:514` and consumed by `trial.suggest_int('window_size', …)`
at `:420-422`. Ordering verified (mutation precedes study creation at `:537`), mutability
verified (unfrozen dataclass), and confirmed **non-conflicting** with the S172 `window_size.min = 6`
ruling — `max(min+1, …)` at `:512` makes inversion structurally impossible.

**Rules B and C are ADVISORY-BY-DESIGN, not dropped wires.** Three independent citations: the
in-code rationale (`window_optimizer_bayesian.py:491-494`), a governing ruling
(`SESSION_CHANGELOG_20260307_S122.md:56` — *"disabled per TB + S121 shuffle test"*), and the
producer's own docstring. **The v1.15 spec text describing them as applied is SUPERSEDED.**
This is the classification the brief specifically warned against getting wrong, and it was
made correctly. **These are not §2.7 instances.**

Ten outputs verified correctly wired (audit §9, VIR-2 clean control).

**The defects are in how Step 0 is launched and how its artifact ages — not in what it
computes.**

## 2. The two live defects, and why they interlock

**F1 — CRITICAL.** `agent_manifests/trse.json` `default_params` declares `window_size`,
`stride` and `trse_context`. WATCHER emits every `default_params` key as `--key value`
(`agents/watcher_agent.py:1496-1512`; no `args_map`, so the underscore→hyphen fallback at
`:1499` applies). **`trse_step0.py` defines none of the three.** Proven by execution this
session, from an empty scratchpad CWD so the repo could not be written:

```
trse_step0.py: error: unrecognized arguments: --window-size 400 --stride 50 --trse-context …
EXIT_CODE=2
```

Masked by `"skip_on_fail": true`, which records the failure as `action: proceed`.
**`watcher_decisions.jsonl` holds 17 such entries**, all reading *"Step 0 output absent —
downstream step runs with defaults."*

**F2 — HIGH.** Step 1 reopens `trse_context.json` in `"w"` at `window_optimizer.py:793-794`
to append `confirmed_windows`. That **bumps the mtime**, which is exactly the sentinel
`check_output_freshness` compares against `daily3.json`
(`agents/watcher_agent.py:495-524`). Live proof: internal `timestamp` **2026-03-13T12:10:27Z**,
file mtime **2026-05-01 16:05:11**, and `confirmed_windows[-1].timestamp` matching that mtime
**to the microsecond**. The regime analysis in force is ~4.5 months old.

**The interlock is the actual finding:**

```
new draws land → Step 0 goes stale → executes → F1 exits 2
  → skip_on_fail proceeds → Step 1 loads the OLD context → applies Rule A from it
  → Step 1 writes back → mtime now exceeds the new daily3.json
  → Step 0 reports "fresh" forever after
```

F1 fires **only** when new draws have arrived — precisely when Step 0 would do useful work.
It is dormant today only because the freshness check short-circuits before the broken command
is ever built. **This is a self-perpetuating lock, and it is currently engaged.**

Risk is bounded but real: Rule A only narrows `max` to 32 and cannot violate `min`, so the
regime input is *unverified* rather than *unsafe*.

## 3. Proposed fix — Alpha's recommendation, for Beta's ruling

Alpha proposes **three changes plus one deliberate non-change.** All are small; the value is
in the ordering and in what is *not* done.

### 3.1 F1 — reconcile the manifest with the CLI, in the manifest

Two candidate directions:

- **(a) Fix the manifest** — remove `window_size`, `stride`, `trse_context` from
  `default_params`, or add an `args_map` mapping them to flags that exist.
- **(b) Fix the script** — add the three arguments to `trse_step0.py`'s parser.

**Alpha recommends (a).** Evidence: a fresh clone has **no** `agent_manifests/trse.json`
(F5), so WATCHER would invoke `python3 trse_step0.py` with no params — **and that succeeds.**
The script's own CLI is therefore the correct contract; the manifest is the artifact that
drifted. Adding arguments to the script to satisfy a malformed manifest would ratify the
drift.

**Caveat requiring Beta's view:** `window_size` and `stride` are real TRSE concepts (the
audit shows sliding windows at W200/W400/W800). If the manifest's intent was to make them
operator-settable, then (a) *removes a capability someone wanted* — the §0.4 hazard. Alpha
found no evidence they were ever accepted by the script, but requests Beta rule on whether
they should be **wired in** (added to the parser and threaded to the windowing code) rather
than removed from the manifest.

### 3.2 F2 — stop Step 1 from owning Step 0's freshness sentinel

The root cause is that **one file serves two owners**. Alpha proposes separating them:

- **Preferred: Step 1 stops writing into `trse_context.json`.** `confirmed_windows` moves to
  its own artifact (e.g. `trse_confirmed_windows.json`). `trse_context.json` becomes
  **write-once by Step 0**, so its mtime means what the freshness check assumes it means.
- **Alternative if the shared file must persist:** change the freshness check to compare the
  **internal `timestamp` field**, not the file mtime — the audit shows the internal value is
  correct and the mtime is not.

**Additionally, and independently: add a staleness guard to `_load_trse_context`**
(`window_optimizer_bayesian.py:25-47`). It has a version guard but **no staleness guard**, so
a 4.5-month-old context is applied unconditionally and silently. Alpha recommends it **warn
loudly and record the context's age in provenance**, and requests Beta rule on whether it
should **fail closed** beyond some age. Alpha does not propose a threshold value — that is a
judgement about how fast regimes drift, which Alpha cannot source.

### 3.3 F3 — make `save_context` merge, before Step 0 can run again

`save_context` (`trse_step0.py:795-800`) writes the fresh context **without merging**, so a
successful Step 0 run would **erase the 20 accumulated `confirmed_windows` entries**
(2026-03-14 → 2026-05-01). This is latent *only* because F1/F2 prevent Step 0 from running.

**Fixing F1 and F2 makes it live.** If `confirmed_windows` moves to its own artifact (3.2
preferred), the destruction path disappears. If not, `save_context` must merge rather than
overwrite. **Alpha flags this as an ordering constraint: F3 must be handled in the same pass
as F1/F2, not after.**

### 3.4 Deliberate non-changes

- **F4 — do not touch.** `consistent_with_known_skip` is a near-vacuous overlap test
  (`gap_range [27, 773]` vs known `[5, 56]` → `true`). **No consequence today** because Rule
  B is disabled. But **hybrid skip bounds are the next approved task**, and anyone
  re-enabling Rule B while wiring them would import a false-positive gate. Alpha proposes
  recording this as a **blocking prerequisite on the skip-bound deliverable** rather than
  fixing it here — changing a disabled rule's gate is out of scope for a launch-path repair.
- **F6 — do not silently remove.** `recommended_window_size` is echoed, loaded into `_rec_ws`
  (`window_optimizer_bayesian.py:500`) and never referenced; Rule A uses a hardcoded `32`
  (`:513`) instead of the planned `rec_ws * 4` (`PLAN:94`). Per §0.4 the disposition is
  **wire-in or explicit retirement by ruling — not silent removal.** Requesting Beta's ruling.

### 3.5 F5 — tracking

`agent_manifests/trse.json` is untracked (`.gitignore:41:*.json`) while the other six
manifests are force-added. So **the live host and the repository disagree about Step 0's
configuration**, no repo gate can see the file that causes F1, and a fresh clone raises in
`get_step_io_from_manifest(step=0)`. Alpha recommends force-adding it, which also gives F1's
fix a reviewable history.

## 4. Sequencing note

If F1 is fixed and F2 is not, Step 0 will begin **running** against a sentinel that still
says "fresh" — so it will run rarely and unpredictably. If F2 is fixed and F1 is not, Step 0
will begin **failing** (exit 2) every time new draws land, masked by `skip_on_fail`. **They
must be fixed together**, with F3 in the same pass per 3.3.

## 5. Rulings requested

1. **F1 direction** — fix the manifest (Alpha's recommendation) or wire `window_size` /
   `stride` into the script as genuine operator controls?
2. **F2 approach** — separate artifacts (preferred) or internal-timestamp freshness check?
3. **Staleness guard** — warn-and-record, or fail closed beyond an age? If the latter, what
   age? Alpha has no basis to propose a number.
4. **F4** — accept as a blocking prerequisite on the hybrid skip-bound deliverable?
5. **F6** — wire in (`rec_ws * 4` per the integration plan) or retire by explicit ruling?
6. **F5** — force-add `agent_manifests/trse.json` to tracking?

## 6. VIR note

The audit's fault-injection control was **n/a for a read-only audit, stated rather than
omitted**. The one positive control available without mutating state was the F1 argparse
reproduction — a genuine executed failure, not an inference. VIR-6 scope: repo + VM 101 only;
the rigs were powered off and were not inspected.
