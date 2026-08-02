# TEAM ALPHA → TEAM BETA — one ruling request on 6-P2 REV3 §2.4

**Re:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md` REV3.

Everything else in REV3 is implementable as written. **This is the one item Alpha will not decide,
and it blocks publication one rather than some later run.**

---

## 1. The situation, measured

`daily3.json` on VM101, this session:

| fact | measured |
|---|---|
| records | **18,068** |
| span | `2000-01-01 evening` → `2026-02-26 midday` |
| stored order vs canonical `(date asc, evening<midday)` | **MATCHES CANONICAL** |
| sessions present | **exactly** `{evening, midday}` |
| **`2026-02-26`** | **midday ONLY — evening ABSENT** |

**The published dataset ends mid-day.** Under Beta's bound ordering, the next scrape's
`2026-02-26 evening` sorts **before** the existing terminal record.

**So publication one is a backfill, not an append**, and under REV3 §2.3 it halts with
`NON_APPEND_INSERTION_REQUIRED` on its first real use.

## 2. Why §2.3's day-boundary rule does not resolve it

Alpha added a rule — *publish through the last complete day, hold back a terminal partial day* — so
that the halt signals genuine anomalies instead of firing on every ordinary run.

**That rule governs what P2 publishes going forward. It cannot retroactively repair a version
already published with a partial trailing day.** The inherited state is the problem, not the policy.

## 3. The two options

**(a) Halt.** The first publication stops with `NON_APPEND_INSERTION_REQUIRED`, reports the
insertion key and position, and a human makes a one-time lineage decision.
*Conservative; keeps the wall absolute on its first real test; costs one manual intervention.*

**(b) Retroactive boundary in the merge.** The merge treats `2026-02-26` as pending on **both**
sides, so the comparison runs to the last complete day and the sequence rejoins at a clean day
boundary — after which every run is a genuine append.
*Less disruptive; no manual step; but the first exercise of the append-only wall is one where the
wall was deliberately routed around.*

## 4. What Alpha has and has not done

- **Implemented:** neither. REV3 gates §2.3's rule and the halt, and leaves the transition
  unimplemented pending this ruling.
- **Recommended:** neither. Alpha can argue both and does not think the choice is Alpha's — it
  trades a procedural cost against the strength of a wall's first test, which is a governance
  judgement.
- **Flagged:** that this is a **one-time transition, not a policy.** Whichever way it goes, §2.3
  governs every subsequent run and no further transition arises.

## 5. A related fact Beta may want to weigh

The dataset is **about five months stale** — last record `2026-02-26`, today `2026-08-02` — because
the scraper has not run (the ENOENT service loop). **The first publication will therefore be large,
not a one-record append.** That does not change the ordering question, but it means option (a)'s
manual step lands on a substantial merge rather than a trivial one, and option (b)'s routed-around
wall would be routed around on a substantial merge too.

**Alpha raises it because it cuts both ways and neither direction is obviously lighter.**

## 6. VIR declaration

- **audit claim scope:** §1's figures are **measured on VM101 this session**, not repo-derived —
  `daily3.json` is gitignored and invisible to any repo-scoped check.
- **searched surfaces:** the live `daily3.json` on VM101; the tracked repo; ser8's two scraper
  copies via Michael.
- **unavailable surfaces:** what the source site currently publishes for `2026-02-26 evening` —
  **only a live scrape establishes that**, and no gate in REV3 depends on it. If that record does
  not exist at source, the backfill case does not fire on publication one — **but the terminal
  partial day remains, so the question stands regardless.**
