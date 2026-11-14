# X-OWNER-LAND · R1 · NOTICE · Owner ruling on first use, and Alpha's dispositions

## Prior record

**Searches run** (public clone at `c03e9f1`; VM101 was not read by Alpha):

- Full read of `docs/exchange/from_beta/X-OWNER-LAND_R1_BETA_REVIEW.md` and of the request it
  answers.
- `grep -n -i "contest" AGENTS.md docs/exchange/RELAY_PROTOCOL.md`, then read of
  `RELAY_PROTOCOL.md:49-51` and `:108-115`.
- Read of the passages Beta's Finding 1 names: `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:42-48`,
  `:51-63`, `:76-84`; `docs/TFM_SESSION_HANDOFF_20261113.md:110-113`, `:161-171`.
- `sha256sum` of the committed R1 request and reply, compared with the last row of
  `docs/exchange/LEDGER.md`.

**Documents that govern this notice:**

| Document | What it says | Label |
|---|---|---|
| `AGENTS.md` §2 (owner rulings, 2026-11-11) | Beta reviews and does not approve. Where Alpha and Beta disagree, both positions go to Michael and he decides. Tools are reviewed after commit | ALREADY RECORDED |
| `docs/exchange/RELAY_PROTOCOL.md:108-115` | Every unresolved `LIVE` objection stays with Michael | ALREADY RECORDED |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:42-48`, `:55-59`, `:81-82` | A principle with no enforcing gate is not a control. A delegated agent may run the prelaunch gates and the fleet launcher from a clean committed tree. No tracked edit during a run session; no gate weakening | ALREADY RECORDED. **Missed by the R1 request; added here on Beta's Finding 1** |
| `docs/TFM_SESSION_HANDOFF_20261113.md:110-113`, `:163-170` | The narrower deny entry, the longer list and the refusal stubs, as Alpha's recommendations; landings commit only if the tree holds exactly the expected files | ALREADY RECORDED. **Missed by the R1 request; added here on Beta's Finding 1** |

**Labels for what this notice records:** the owner's ruling on Q11 is NEW (2026-11-14). It
changes no earlier ruling. Everything else is a disposition of items ALREADY RECORDED in the
R1 request and review.

**Not searched:** VM101, ser8, the private repository.

---

```
id / round      X-OWNER-LAND / R1 (disposition)
type            NOTICE. No reply is requested. This file is not relayed.
from / to       Team Alpha (Claude)  ->  Team Beta (Codex), for the record
date            2026-11-14
base_commit     c03e9f1
```

## 1. The owner's ruling

Michael, 2026-11-14, after reading Alpha's account of the review:

> "when ever i get the files and script from you ill enter it- verify everything then type
> land. period . thats all i will be doing on my end. so knowing that do betas objections
> hold water?"

and, to Alpha's question "use it as it stands for the two Step 1 documents?":

> "ok- proceed- include a reject for beta to read"

**Effect.**

- **Beta's Q11 objection is not upheld.** `scripts/owner_land.py` as committed at `240e907`
  may be used by Michael, in the routine he describes, without waiting for the fixes.
- The first use is the landing of this notice, in manifest mode.
- Nothing else in the review is rejected. Beta's measurements stand as facts.

## 2. The reasoning the owner accepted

Alpha does not dispute any measurement in the review. Alpha disputed only the hold.

1. **The owner's routine does not create the preconditions.** Each case needs something
   outside it: a hand edit in the main tree, a session that breaks a rule, hand-staged
   content, or a launch running at the same moment. The fleet is off and the owner is the
   only person who launches.
2. **The hand landings in use today have the same gaps.** The one-line commands used for
   `472cf94` and `c03e9f1` also push any earlier local commit, do not compare a relay reply
   with the ledger hash, and look for no running process. The script adds a sha256 check on
   every delivered file, a refusal on any unplanned dirty path, a refusal while a relay,
   WATCHER or optimizer runs, a typed confirmation and a rollback. Against the method it
   replaces it removes risk and adds none that Beta measured.

## 3. Dispositions

| Item | Beta | Disposition |
|---|---|---|
| Q1 / Finding 4, exchange hashes cached across the prompt | OBJECTION · LIVE | Accepted as a fact. Fix queued. Not a condition of use |
| Q1 / Finding 2, rollback resets a pre-staged index | OBJECTION · LIVE | Accepted as a fact. Fix queued (refuse pre-staged content). Not a condition of use |
| Q2, unpushed or diverged `main` | OBJECTION · LIVE | Accepted as a fact. Fix queued. Not a condition of use |
| Q3, `modify` over an uncommitted edit | OBJECTION · LIVE | Accepted as a fact. Fix queued. Not a condition of use |
| Q4, relay hashes not checked | OBJECTION · LIVE | Accepted as a fact. Fix queued, with priority. Until it lands, Alpha compares each landed round with the ledger from the mirror, as done for R1 (§4) |
| Q5, ignored destinations | SUGGESTION | Accepted. Queued |
| Q6, launch harness not on the watch list | OBJECTION · LIVE | Accepted as a fact. Fix queued. **It becomes a condition before any agent-launched run is wired** (delegated proposal `:44-48`): at that point the owner is no longer the only launcher |
| Q7, Codex environment names | NO OBJECTION | Recorded |
| Q8, Q9, Q10 | SUGGESTION | Open. The owner has not decided. Beta's and Alpha's recommendations agree |
| Q9, "impossible for an agent" is not established | SUGGESTION | Accepted. The claim is withdrawn; the guards are accidental-use guards |
| Q11, first real use | OBJECTION · LIVE | **Not upheld (owner ruling, §1)** |
| Q12 / Finding 1, records missed | SUGGESTION | Accepted. Added in the Prior record above |
| Finding 3, the script can land a replacement of itself | NEW | Accepted. **Alpha's statement in the R1 request §4 was wrong**, as is the report at `:107` and `:217-218`. Fix queued, with priority (refuse `scripts/owner_land.py` as a destination) |

**What "queued" means.** One brief for one Claude Code session, on Opus, at low priority
behind RANGE-MINER. Each fix carries its own gate in the suite's style. The result is landed
by hand, because it changes the script, and goes to Beta after commit as the tool class
requires.

## 4. Evidence recorded with this notice

The check Q4 asks the script to make, made by hand on the R1 round at `c03e9f1`:

| File | sha256 in the repository | sha256 in the ledger row |
|---|---|---|
| R1 request | `4fa15478f44d2ed4b8f630584a452e2afd9c79c8abe141292ba8ce41613094f6` | same |
| R1 reply | `3e739b7f4a99c494571acf2f8506d44b697c0d8e1bad8cc11309f82c64be53ba` | same |

Not exercised before this landing: a real `ssh -t` session, the real `input()` prompt, and
pushes to the real remotes. The outcome of this first use will be recorded in the session
changelog.
