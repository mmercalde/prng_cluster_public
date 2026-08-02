# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md — REV4 (DRAFT — pending Beta)

**S172 — Phase 6-P2: append-only immutable dataset publication.**

**Pre-implementation commit: `07a2032661cdb3078358cd143838cf1eba0d32d9`** (`07a2032`).
Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.
**Authority:** `docs/DATASET_PUBLICATION_SCHEMA_v1.md` (FROZEN, `manifest_schema_version: 1`).
**Where this brief and the schema differ, the schema wins.**

---

## 0. REV3 → REV4

Beta ruled the transition (§2) and returned four corrections. **Correction 1 is a logic error in
Alpha's own terminal-day rule, not a tightening of it** — see §3.1. Everything else in REV3 was
ratified.

---

## 1. Verified state of the real dataset — measured on VM101

| fact | measured |
|---|---|
| records | **18,068** |
| span | `2000-01-01 evening` → `2026-02-26 midday` |
| sessions present | **exactly** `{evening, midday}` |
| stored order vs canonical | **MATCHES CANONICAL** |
| `2026-02-26` | **midday only — evening ABSENT** |
| single-session dates | **1,040 of 9,554** — **1,038 are 2000-2002** (evening-only era); 2019: **1**; 2026: **1** |

`2019-01-25` evening-only is a separate anomaly → BACKLOG, **not P2's work**.

---

## 2. THE TRANSITION — Beta's binding ruling: option (a), fail closed

**Option (b) is REJECTED.** It cannot be characterized as "rejoining at a day boundary": it would
**insert a record before the already-published `2026-02-26 midday`**, so L001's record sequence
would **no longer be a prefix**. That directly violates the frozen lineage contract.

**If a live scrape supplies `2026-02-26 evening`, publication against L001 MUST halt with
`NON_APPEND_INSERTION_REQUIRED`:**

- **preserve L001 byte-for-byte;**
- **write no version and no pointer;**
- **report every insertion key and canonical position;**
- **do NOT autonomously create L002;**
- the later live-scrape **dry run** determines the exact contents of a **separately authorized**
  new-lineage transition.

**If the source does not contain that evening record, no transition exception fires** and ordinary
merge rules govern the later records.

**G-VERSION-ONE-CASE (required):** existing terminal `2026-02-26 midday` **plus** scraped
`2026-02-26 evening` **must halt**. Gate this exact case.

---

## 3. Canonical order, deferral, and backfills

### 3.0 The order — bound

```
(date ascending, session precedence: evening BEFORE midday)
```

Allowed session values are **exactly** `evening` and `midday`. An unknown value **fails
validation** — it never acquires an invented ordering. JSON whitespace and object-key order are
irrelevant; **array record order is semantic.**

**Never sort the prior version.** The current version must already equal its canonical ordering or
publication **halts as invalid authority state**. *(Measured today: it does. The check still runs.)*

### 3.1 Terminal midday-only deferral — CORRECTED

**Alpha's REV3 rule was wrong.** It said to hold the final date when it *"carries a midday
record."* Under the bound order **every complete date ends with its midday record**, so that rule
would **defer every complete terminal date indefinitely** and the dataset would never advance.

**Replace it with an exact post-dedup session-set rule on the terminal date:**

| terminal-date sessions | behaviour |
|---|---|
| `{evening, midday}` | **publish normally** |
| `{midday}` | **defer that date** |
| `{evening}` | **publish normally** — a later midday remains an **append** under canonical order |
| anything else | **validation failure** |

**Call this a terminal midday-only deferral.** It is **not** a claim that the day is physically
complete or incomplete — only that a midday-without-evening trailing edge is the shape that would
force a backfill next run.

**A deferral produces a structured result DISTINCT from ordinary `NO_CHANGE`, carrying the deferred
keys:**

- if **other publishable additions exist** → **publish them** and **report** the deferred terminal
  date;
- if **nothing else changed** → **write nothing** and return **`DEFERRED`**.

**Note this rule does not rescue publication one.** The *existing published* version ends
`{midday}`-only; deferral governs what P2 publishes going forward, so §2's halt still applies.
The two are independent.

### 3.2 Backfill halt

**A new record sorting INSIDE the prior sequence is a backfill, not an append** →
**`NON_APPEND_INSERTION_REQUIRED`**: no version written · pointer unchanged · no autonomous lineage
· **report insertion keys and canonical positions**.

---

## 4. `--recent` must not lose a deferred record across a calendar boundary

**The defect:** `--recent` sets `start_year = end_year = TODAY.year`. A record deferred on
**December 31** is then **never re-fetched** in January — the deferral would silently lose it.

**Required, for `--recent`:** derive the scrape range **from the validated current authority**:

```
start_year = year(current_manifest_target.last_draw.date)
end_year   = current year
```

**Fetch every year in that inclusive range.** This preserves overlap with the last published
boundary **and** permits catch-up after extended scraper downtime.

**⚠ The pre-fetch authority read selects the scrape range ONLY.** Publication must still
**reacquire the writer lock and revalidate the authority before merging** (§6). The two reads are
separate and the second is authoritative.

**G-CALENDAR-BOUNDARY (required):** a record deferred on **31 December** is **re-fetched in
January and subsequently published.**

---

## 5. The merge

**Inputs:** the validated current manifest and its immutable target; **and** newly scraped records,
**which may be only a partial year**.

**Draw identity is `(date, session)`.**

| case | behaviour |
|---|---|
| duplicate key, **identical** draw | **collapse idempotently** |
| two scraped records, same key, **different** draws | halt **`SOURCE_CONFLICT`** |
| scraped record conflicts with an **already-published** key | halt **`CORRECTION_REQUIRED`** |
| published key **absent** from a partial scrape | **retained** |
| previously unseen key | **added** |
| **any** missing scrape record | **never** a deletion |

**No-change:** if the merged canonical sequence **equals** the current sequence → write **nothing**
— no version, no pointer update, no timestamp-only generation, no predecessor link.

---

## 6. Durable publication — settled primitives (Beta correction 3)

**REV3 left two choices to the implementer. They are settled here.**

- **Install immutable versions with same-filesystem `os.link(temp, final)`, then unlink the temp.**
- **Do NOT use `open(final, "x")` to copy bytes into a visible final name** — that can **expose a
  partial final file**.
- **Hold `fcntl.flock(LOCK_EX)` on an opened publication-root DIRECTORY**, from **before the
  pointer is read** through **the final directory fsync**, **including no-change and failure
  cleanup paths.**

**The transaction, in binding order:**

```
 1. open the publication-root directory; fcntl.flock(LOCK_EX)
 2. read + validate the pointer and its target (§7)
 3. merge (§5), apply §3.1 deferral and §3.2 / §2 halts
 4. serialize deterministically; derive the digest from the EXACT BYTES
 5. write the version to a same-directory temporary .json
 6. flush + fsync it
 7. validate it FROM DISK
 8. os.link(temp, final); unlink temp        <- no-clobber install
 9. fsync the directory
10. write + flush + fsync the manifest temporary
11. os.replace the pointer; fsync the directory again
12. release the lock (also on every no-change and failure path)
```

**`os.replace` is correct for the mutable pointer and FORBIDDEN for the immutable version install**
— it can overwrite an existing artifact. **The lock must not require a permanent non-`.json`
sidecar**; a directory advisory lock satisfies this.

### 6.1 Crash recovery

Between steps 8 and 11 a **complete version exists while the pointer still names its predecessor.**

On retry: an **exactly matching, fully validated orphan may be ADOPTED**; it must **never be
overwritten**. **Multiple ambiguous matching candidates fail closed.**

### 6.2 Concurrency — a real two-process gate

**Two publishers starting from the same predecessor must serialize, re-read authority under the
lock, and produce a valid single chain — never two competing successors.** This is a **real
two-process test**, not a simulated one.

---

## 7. Current-authority validation, and evaluation precedence (Beta correction 4)

### 7.1 Authority source

**All operational dataset facts derive from `daily3_current.json` and its validated target.**
**The frozen `daily3.json` may be read ONLY by `G-ALIAS-FROZEN`.**

### 7.2 Failure precedence — binding order

```
1. current-authority validation
2. scraped-record validation
3. intra-scrape SOURCE_CONFLICT
4. published-key CORRECTION_REQUIRED
5. NON_APPEND_INSERTION_REQUIRED
6. terminal midday-only deferral
7. no-change
8. publication
```

### 7.3 Authority validation — individually named gates

**REV3's vague "twelve authority failures" is replaced.** One named gate per condition:

`G-AUTH-POINTER-MISSING` · `G-AUTH-POINTER-MALFORMED` · `G-AUTH-SCHEMA-MISMATCH` ·
`G-AUTH-FORBIDDEN-TARGET-NAME` · `G-AUTH-TARGET-MISSING` · `G-AUTH-FILENAME-VERSIONID-DISAGREE` ·
`G-AUTH-SIZE-MISMATCH` · `G-AUTH-RECORD-COUNT-MISMATCH` · **`G-AUTH-FIRST-RECORD-MISMATCH`** ·
**`G-AUTH-LAST-RECORD-MISMATCH`** · `G-AUTH-DIGEST-MISMATCH` · `G-AUTH-FILENAME-PREFIX-MISMATCH` ·
`G-AUTH-INVALID-LINEAGE-ID` · `G-AUTH-NONCANONICAL-PRIOR-ORDER`

**First-record and last-record mismatches are exercised INDEPENDENTLY** — not as one combined case.

**Mutants:** a corrupted pointer · **a valid-JSON target with one changed digit**.

Run the **frozen publication validation authority**, or **extract and reuse it. Never create a
weaker duplicate.**

---

## 8. `daily3.json` — PERMANENTLY FROZEN

**Do not refresh it. Ever.** P0.5 made `daily3_current.json` authoritative and classified
`daily3.json` a **legacy compatibility alias**. Consumers still opening it remain **intentionally
stale** until separately migrated.

**G-ALIAS-FROZEN:** bytes, digest, size and mtime unchanged across **all five** of: successful
append · no-change publication · correction halt · injected crash · malformed
current-publication state.

**G-TREE-CLEAN:** either compare porcelain **before vs after** and prove publication added nothing,
**or** operate in a **temporary git repository carrying the real ignore rules**. **The `.sha256`
mutant must produce the ONLY new porcelain entry.**

---

## 9. Source ownership and deployment

| item | value |
|---|---|
| canonical scraper, tracked | **`daily3_scraper.py`** at repo root |
| publication module, tracked | **`utils/dataset_publication.py`** — must not import from `miner/` |
| Revision 1.5 baseline | record the **sha256** of the retained source in report and commit message |
| ser8 copies | **RETIRED as authorities** — historical artifacts, not deployment targets |
| invocation | `python3 daily3_scraper.py --json --recent` from the repo root |

**Stop condition:** a **divergent** copy on VM101 or Zeus → **STOP and report** (second lineage).
*(ser8's two copies are byte-identical Revision 1.5, confirmed.)*

## 10. Service — separate work, NOT here

**Do not repoint the current unit at the real scraper.** A terminating scraper under
`Restart=always` **executes continuously and hammers the source site.**

After certification: one-shot service **plus timer** (or an explicitly scheduled wrapper) ·
real-scrape **dry run with no publication** · resolve any correction/backfill halt · **one
controlled publication** · post-publication verifier **PASS** · **only then** enablement.

## 11. Do not "fix" the 22 zero-draw records

`draw == 0` is **genuine data** — `000` is legitimate; 22 sits inside the observed range 4-32
(median 18, expected 18.07); indices 254-16,939, not clustered at an import boundary. The defect is
in two consumers **off the certifying path**, both using `entry.get("draw") or …`:
`digit_sequential_sieve.py:161-162`, `coordinator.py:1881`. The certifying loader is zero-safe
(`miner/range_miner_worker.py:575`). `DAILY3_CONSUMER_CONTRACT_v1.md` §9 MUST #5 requires `0` be
emitted as `0`. **Never bundled with a data publication.**

## 12. Out of scope

Session splits (unversioned, unbound, deliberate open item) · the zero-droppers · the certifying
Step-1 path · `2019-01-25` (BACKLOG) · service activation (§10) · **creating L002** (§2).

---

## 13. Gates — `tests/test_s172_phase_6_p2_publication.py`

**Every gate uses a temporary publication root. No gate touches the real `daily3.json` or the
network.**

**Transition:** **`G-VERSION-ONE-CASE`** (§2) — terminal `2026-02-26 midday` + scraped
`2026-02-26 evening` → **halt**, L001 byte-identical, no version, no pointer, no L002, insertion
keys and positions reported.

**Deferral (§3.1):** `{evening, midday}` → publish · **`{midday}` → defer** · `{evening}` →
publish, and a later midday **appends** · unknown session → **validation failure** · deferral with
other additions → **publish those + report deferred** · deferral alone → **write nothing, return
`DEFERRED`** · **`DEFERRED` is distinguishable from `NO_CHANGE`.**

**Calendar (§4):** **`G-CALENDAR-BOUNDARY`** — deferred 31 Dec re-fetched in January and published ·
scrape range derived from authority `last_draw.date` · **the pre-fetch read does not substitute for
the under-lock revalidation.**

**Merge (§5):** identical repeated scrape → nothing written · subset-only → no-change · subset + one
new record → one version · exact duplicates collapse · intra-scrape conflict → `SOURCE_CONFLICT` ·
published conflict → `CORRECTION_REQUIRED`, nothing written, no L002 · absent published key →
**retained**.

**Order (§3):** canonical order · prior **not sorted**; non-canonical prior **halts** · backfill →
`NON_APPEND_INSERTION_REQUIRED` · **`G-HISTORICAL-SINGLE`:** an **interior** evening-only date
(2000-2002 shape) publishes normally and is **never** treated as partial.

**Authority (§7.3):** the fourteen named gates, first/last record **independent**.

**Precedence (§7.2):** construct inputs triggering **two** conditions at once and prove the
**earlier** one wins. At minimum: authority-invalid + source-conflict → authority ·
correction + insertion → correction · insertion + deferral → insertion.

**Durability (§6):** the twelve-step order · **`os.link` refuses a pre-existing path** ·
**`open(final,"x")` byte-copy is absent** (AST) · lock held across **no-change and failure** paths ·
five crash gates (partial temp write · complete version pre-install · installed pre-pointer ·
manifest temp pre-replace · pointer replaced pre-final-fsync) · **orphan adoption**: exact match
adopted, never overwritten; **ambiguous candidates fail closed** · **`G-CONCURRENCY`: two real
processes from one predecessor → serialized, single valid chain.**

**Identity:** `filename == version_id + ".json"` · 12-hex prefix matches the digest's first 12 ·
`record_count`, `size_bytes`, `first_draw`, `last_draw` correct · `predecessor_sha256` chains,
`null` on a lineage's first.

**§8:** `G-ALIAS-FROZEN` five scenarios · `G-TREE-CLEAN` corrected form · **`G-ZERO-DRAWS`**.

**Mutants:** restore the overwrite-with-subset path · byte-prefix test (**must red on a
reformatting case**) · **REV3's broken "carries a midday record" predicate** (must red by deferring
a complete `{evening,midday}` terminal date) · `--recent` reverts to current-year-only (**must red
G-CALENDAR-BOUNDARY**) · derive canonical order from scrape order · sort the prior version ·
rewrite a prior record instead of halting · create L002 autonomously · missing scrape record read as
deletion · `os.replace` for the version install · skip an fsync · release the lock before the final
fsync · adopt a **non**-matching orphan · overwrite an orphan · `.sha256` sidecar (**only new
porcelain entry**) · refresh `daily3.json` · drop `predecessor_sha256` · drop a zero-draw record ·
reorder the §7.2 precedence.

---

## 14. Report

`docs/S172_PHASE_6_P2_IMPLEMENTATION_REPORT.md`: §9's tracked paths and the Revision 1.5 sha256 ·
the VM101/Zeus divergent-copy check · the publication module and its seam · **the corrected
terminal-date session-set predicate and its four rows** · `DEFERRED` vs `NO_CHANGE` distinguishability
· the `--recent` range derivation and the calendar-boundary evidence · the fourteen named authority
gates with first/last independent · the §7.2 precedence evidence · **`os.link` install with its
refusal proof** · the flock scope including failure paths · five crash gates and orphan adoption ·
**the two-process concurrency result** · `G-VERSION-ONE-CASE` · `G-ALIAS-FROZEN` five scenarios ·
gate/mutant counts · confirmation **no gate touched the real dataset or the network**.
Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; crash gates
  report the artifacts present at the injected point; the concurrency gate reports both PIDs and
  the resulting chain.
- **clean control:** a normal append publishes, chains, defers nothing, and leaves the tree clean.
- **fault-injection control:** §13's mutants, four-part kill rule — prove each red comes **from its
  injected defect**.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** no fleet dependency — must pass with all rigs down.
- **audit claim scope:** §1's facts are **measured on VM101**; the scraper characterization is from
  **ser8 Revision 1.5**. Repo pinned at **`07a2032`**.
- **searched surfaces:** tracked repo at `07a2032`; ser8 `~/Downloads/` and `~/cluster_controller/`
  via Michael; the live `daily3.json` on VM101.
- **unavailable surfaces:** whether a **divergent** copy exists on VM101/Zeus (§9 stop condition);
  **what the source currently publishes for `2026-02-26 evening`** — only the §10 dry run
  establishes it, and **no gate may depend on it.**
