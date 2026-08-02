# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md — REV3 (DRAFT — pending Beta)

**S172 — Phase 6-P2: append-only immutable dataset publication.**

**Base:** HEAD (current). Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.
**Authority:** `docs/DATASET_PUBLICATION_SCHEMA_v1.md` (FROZEN, `manifest_schema_version: 1`).
**Where this brief and the schema differ, the schema wins.**

---

## 0. Verified state of the real dataset — measured on VM101 this session

Beta's correction 2 rests on claims about the existing data. **They were measured, not assumed.**

| fact | measured |
|---|---|
| record count | **18,068** |
| span | `2000-01-01 evening` → `2026-02-26 midday` |
| session values present | **exactly** `{evening, midday}` |
| stored order vs canonical `(date asc, evening<midday)` | **MATCHES CANONICAL** |
| `2026-02-26` records | **midday only — evening ABSENT** |
| single-session dates | **1,040 of 9,554** — but **1,038 are 2000-2002** (evening-only era), then 2019: **1**, 2026: **1** |

**Two consequences.**

**The authority state is valid**, so publication does not halt under Beta correction 3's ordering
wall. Good — but it must still be *checked*, not assumed, because that is what §3 is for.

**Beta's backfill case fires on publication one.** The dataset ends `2026-02-26 midday`; the next
scrape finds `2026-02-26 evening`, which sorts **before** it. That is a **backfill, not an
append**.

**And it is structural, not a one-off** — any scrape ending on a midday record guarantees the next
run backfills. §2.3 addresses that. `2019-01-25` evening-only is a separate anomaly, recorded in
BACKLOG, **not P2's work**.

---

## 1. ⚠ The defect being removed

`main()` accumulates into `all_draws`, then:

```python
Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))   # OUTPUT_FILE = "daily3.json"
```

**It never reads the existing dataset.** With `--recent` (`start_year = end_year = TODAY.year`),
**`daily3_scraper.py --json --recent` replaces 18,068 records with the current year alone.** No
merge, no dedup, no error, exit status success. `write_text` is not atomic, so a crash mid-write
truncates outright.

`daily3scraper.service` — enabled since Sep 2025, `Restart=always`, targeting a
`run_daily3scraper.py` that **never existed** — has looped ENOENT every boot. **That loop is the
only thing that has prevented this.** See §7.

## 1.1 Current behaviour, verified at source (ser8 Revision 1.5)

Full overwrite · never reads prior data · **no dedup** (the PA descendant's header claims dedup;
the CA original has none) · **no sort** · no versions, lineage, manifest or correction concept ·
emits `date`/`session`/`draw` · `int("000")` → `0`, **correct** per the consumer contract.

## 1.2 Scope of change

**Keep:** `fetch_draws`, parsing, row filtering, CLI, the year loop. Not the defect, not revised.
**Replace entirely:** everything from `all_draws` to disk.
**Structure the publication as a separate module** so the contract is testable without the network.

---

## 2. Canonical order, backfills, and the day boundary

### 2.1 The order — bound

```
(date ascending, session precedence: evening BEFORE midday)
```

**Allowed session values are exactly `evening` and `midday`.** An unknown value **fails
validation** — it never acquires an invented ordering.

**JSON whitespace and object-key order are irrelevant. Array record order is semantic.**

### 2.2 Never sort the prior version

**Do not sort the current published version.** Sorting it would silently hide a non-canonical
published sequence. **The current version must already equal its canonical ordering, or
publication halts as invalid authority state.**

*(Measured today: it does. The check must still run.)*

### 2.3 Backfill halt — and the day-boundary rule that keeps it meaningful

**A new record that sorts INSIDE the prior sequence is a backfill, not an append.** It halts with
**`NON_APPEND_INSERTION_REQUIRED`**: no version written · pointer unchanged · no autonomous lineage
· **report the insertion keys and positions for human lineage review.**

**Alpha's addition — the terminal partial day.** Without this the halt fires on ordinary runs, and
a scraper that needs a human decision most times it runs is not usable.

> **Publish through the last COMPLETE day. Hold back a terminal partial day.**
>
> If the final date in the merged sequence carries a **midday** record, its **evening** record may
> still be pending at the source. **Drop that date's records from this publication** and pick them
> up on the next run. Every publication then ends on a day boundary, so every subsequent scrape is
> a genuine append, and `NON_APPEND_INSERTION_REQUIRED` becomes an **anomaly signal** rather than a
> routine event.

**Why the rule is about the TERMINAL record and not about "both sessions present."** A
"both sessions required" rule would reject **1,038 legitimate 2000-2002 evening-only dates**. Those
dates predate CA Daily 3's midday draw. **They are complete.** The rule applies only to the trailing
edge, where a session may still be pending.

**A held-back date is not a deletion and not a conflict.** It is deferred. Say so in the report and
log it.

### 2.4 ⚠ TRANSITION QUESTION — for Beta, NOT for the implementer

**The existing published version already ends at a partial day** (`2026-02-26 midday`, evening
absent). So publication one inherits the problem regardless of §2.3, which governs only what P2
publishes going forward.

Two options, and **Alpha will not choose**:

**(a)** The first publication **halts** with `NON_APPEND_INSERTION_REQUIRED` and a human makes a
one-time lineage decision.
**(b)** The day-boundary rule applies **retroactively when computing the merge**, so the merge
treats `2026-02-26` as pending on both sides and the sequence rejoins cleanly at a day boundary.

**(b) is less disruptive; (a) is more conservative and keeps the wall absolute on its first real
test.** This is a one-time transition, not a policy — **flagged for ruling. Implement neither until
Beta answers.** Until then, gate the §2.3 rule and the halt; leave the transition unimplemented.

---

## 3. Validate the current authority BEFORE merging (Beta correction 3)

Run the **frozen publication validation authority** — or **extract and reuse it. Never create a
weaker duplicate.**

**Fail before writing on any of:** missing or malformed pointer · schema mismatch · forbidden
target name · missing target · filename/version-ID disagreement · size mismatch · record-count
mismatch · first/last-record mismatch · **full digest mismatch** · filename-prefix mismatch ·
invalid lineage identity · **prior sequence not equal to its canonical ordering** (§2.2).

**Mutants:** a corrupted pointer · **a valid-JSON target with one changed digit** (proves the
digest check is real and not a shape check).

---

## 4. The merge (Beta correction 1)

**Inputs:** the validated current manifest and its immutable target; **and** the newly scraped
records, **which may be only a partial year**.

**Draw identity is `(date, session)`.**

| case | behaviour |
|---|---|
| duplicate key, **identical** draw | **collapse idempotently** |
| two scraped records, same key, **different** draws | halt **`SOURCE_CONFLICT`** |
| scraped record conflicts with an **already-published** key | halt **`CORRECTION_REQUIRED`** |
| published key **absent** from a partial scrape | **retained** |
| previously unseen key | **added** |
| **any** missing scrape record | **never** interpreted as a deletion |

**No-change result.** If the merged canonical sequence **equals** the current sequence: return
no-change and **write nothing** — no version file, no pointer update, no timestamp-only generation,
no predecessor link.

**Gates:** identical repeated scrape · subset-only scrape with no new records · subset containing
one valid new record · exact duplicate rows · conflicting duplicates within the scrape.

---

## 5. Durable publication — the full transaction (Beta correction 4)

**Atomic pointer replacement alone is insufficient.** Order is binding:

```
 1. acquire ONE publication-writer lock          (before reading the pointer)
 2. validate (§3) and merge (§4) under that lock
 3. serialize deterministically; derive the digest from the EXACT BYTES written
 4. write the version to a same-directory temporary .json
 5. flush + fsync it
 6. validate it FROM DISK
 7. install the final version name atomically, WITHOUT overwriting any existing path
 8. fsync the directory
 9. write + flush + fsync the manifest temporary
10. os.replace the pointer; fsync the directory again
```

**`os.replace` is correct for the mutable pointer and FORBIDDEN for installing an immutable version
file — it can overwrite an existing artifact.** Use a no-clobber primitive: `os.link(tmp, final)`
then unlink the temp, or `open(final, "x")`. **State which, and prove it raises on a pre-existing
path.**

**The writer lock must not require a permanent non-`.json` sidecar** — that would dirty the tree and
break §6. **A directory advisory lock is acceptable.**

### 5.1 Crash recovery — the unavoidable window

Between steps 7 and 10 a **complete version exists while the pointer still names its predecessor.**

On retry: an **exactly matching, fully validated orphan may be ADOPTED.** It must **never be
overwritten.** **Multiple ambiguous matching candidates fail closed.**

**Crash gates at:** partial temporary-version write · complete version before final-name
installation · final version installed before pointer replacement · manifest temporary write before
replacement · pointer replacement before final directory sync.

**Invariant: every visible final-name version is always complete and digest-valid.**

---

## 6. `daily3.json` — PERMANENTLY FROZEN (Beta ruling, §3.4 now closed)

**Do not refresh it. Ever.** P0.5 already made `daily3_current.json` authoritative and classified
`daily3.json` as a **legacy compatibility alias**. Any consumer still opening it remains
**intentionally stale** until separately migrated.

**G-ALIAS-FROZEN:** the alias's **bytes, digest, size and mtime** are unchanged across **all five**
of: successful append · no-change publication · correction halt · injected crash · malformed
current-publication state.

## 6.1 G-TREE-CLEAN — corrected (Beta correction 6)

The development repo may already hold the uncommitted P2 implementation, so **requiring a literally
empty `git status` is invalid or vacuous.** The gate must either:

- compare porcelain **before vs after** publication and prove **publication added nothing**; or
- operate in a **temporary git repository carrying the real ignore rules**.

**The `.sha256` sidecar mutant must produce the ONLY new porcelain entry.**

---

## 7. Source ownership and deployment (Beta correction 5)

REV2 named two off-repository copies and never named the target. **P2 code must be tracked. An
off-repository edit on ser8 cannot be the sole production implementation.**

REV3 specifies:

| item | value |
|---|---|
| canonical scraper, tracked path | **`daily3_scraper.py`** at repo root — beside `pa_pick3_scraper.py`, which is already tracked there |
| publication module, tracked path | **`utils/dataset_publication.py`** — `utils/` holds the existing frozen authorities (`canonical_arrays`, `canonical_records`, `prng_encoding`, `run_finalizer`) and the dependency direction is right: publication must not import from `miner/` |
| retained Revision 1.5 source | record the **sha256 of `ser8:~/Downloads/daily3_scraper.py`** in the report and in the commit message, so the pre-P2 baseline is identifiable forever |
| ser8 copies | **RETIRED as sources.** `~/Downloads/` and `~/cluster_controller/` become historical artifacts, not deployment targets. The tracked repo file is the only implementation. |
| invocation | operator: `python3 daily3_scraper.py --json --recent` from the repo root. Service activation is **§8, separate work.** |

**Stop condition, unchanged:** if a **divergent** copy of `daily3_scraper.py` exists on VM101 or
Zeus, **STOP and report** — that is a second lineage and a different target. *(ser8's two copies are
byte-identical, Revision 1.5, confirmed.)*

---

## 8. Service sequencing — separate work, do NOT do it here

**Do not re-enable `daily3scraper.service` by repointing its current unit at the real scraper.**
**A terminating scraper under `Restart=always` would execute continuously and hammer the source
site.**

After P2 certification, activation is its own work: a **one-shot service plus timer** (or an
explicitly scheduled long-running wrapper) · a **real-scrape dry run with no publication** ·
resolution of any correction/backfill halt · **one controlled publication** · **post-publication
verifier PASS** · **only then** enablement.

---

## 9. Do not "fix" the 22 zero-draw records

`draw == 0` is **genuine data** — `000` is a legitimate outcome; 22 sits inside the observed
per-value range 4-32 (median 18, expected 18.07); indices span 254 to 16,939, not clustered at an
import boundary.

**The defect is in two consumers, both off the certifying path**, both using `entry.get("draw") or …`:
`digit_sequential_sieve.py:161-162` and `coordinator.py:1881`. The certifying loader is zero-safe
(`miner/range_miner_worker.py:575`). `DAILY3_CONSUMER_CONTRACT_v1.md` §9 MUST #5 requires `0` be
emitted as `0`. **Fixing the droppers is separate, non-certifying, and never bundled with a data
publication.**

## 10. Out of scope

Session splits `daily3_midday.json` / `daily3_evening.json` — **not covered by the frozen schema**,
unversioned and unbound, deliberate open item. The zero-droppers. The certifying Step-1 path. The
`2019-01-25` anomaly (BACKLOG). Service activation (§8). The §2.4 transition (awaiting ruling).

---

## 11. Gates — `tests/test_s172_phase_6_p2_publication.py`

**Every gate uses a temporary publication root. No gate touches the real `daily3.json` or the
network** — feed the publication module fixture records directly.

**Merge (§4):** identical repeated scrape → no-change, **nothing written** · subset-only, no new
records → no-change · subset + one new record → one new version · exact duplicate rows → collapse ·
conflicting duplicates in scrape → **`SOURCE_CONFLICT`** · scraped vs published conflict →
**`CORRECTION_REQUIRED`**, nothing written, **no `L002`** · published key absent from partial scrape
→ **retained, never deleted**.

**Order and backfill (§2):** canonical order is `(date asc, evening<midday)` · unknown session
**fails validation** · prior version **not sorted**; a non-canonical prior **halts** · a record
sorting inside the prior sequence → **`NON_APPEND_INSERTION_REQUIRED`**, keys and positions
reported · **G-DAY-BOUNDARY:** a scrape ending on a midday record **holds that date back**, and the
next run appends both its records cleanly · **G-HISTORICAL-SINGLE:** an evening-only date in the
**interior** (2000-2002 shape) publishes normally and is **never** treated as partial.

**Authority (§3):** all twelve failure modes · mutants: corrupted pointer · **valid-JSON target,
one changed digit**.

**Durability (§5):** the ten-step order · **version install refuses to overwrite an existing path**
· five crash gates · **orphan adoption**: exact match adopted and **never overwritten**; **multiple
ambiguous candidates fail closed** · every visible final-name version complete and digest-valid.

**Identity:** `filename == version_id + ".json"` · 12-hex prefix matches the full digest's first 12
· `record_count`, `size_bytes`, `first_draw`, `last_draw` correct · `predecessor_sha256` chains
within a lineage, `null` on a lineage's first.

**§6:** G-ALIAS-FROZEN across all five scenarios · G-TREE-CLEAN corrected form ·
**G-ZERO-DRAWS:** `draw == 0` survives unmodified.

**Mutants:** restore the overwrite-with-subset path · byte-prefix test (**must red on a
reformatting case**) · derive canonical order from scrape order · sort the prior version · rewrite
a prior record instead of halting · create `L002` autonomously · treat a missing scrape record as a
deletion · publish a terminal partial day · **`os.replace` for the version install** · skip an
fsync · adopt a **non**-matching orphan · overwrite an orphan · write a `.sha256` sidecar
(**the only new porcelain entry**) · refresh `daily3.json` · drop `predecessor_sha256` · drop a
zero-draw record.

---

## 12. Report

`docs/S172_PHASE_6_P2_IMPLEMENTATION_REPORT.md`: the §7 tracked paths and the Revision 1.5 sha256 ·
the VM101/Zeus divergent-copy check · the publication module and its seam · the canonical order and
**proof it is not scrape-derived** · the merge matrix results · **the day-boundary rule's two gates**
· the twelve authority failures · the no-clobber install primitive and its refusal proof · the five
crash gates and orphan-adoption evidence · G-ALIAS-FROZEN across five scenarios · the corrected
G-TREE-CLEAN form · gate/mutant counts · confirmation **no gate touched the real dataset or the
network** · **§2.4 left unimplemented pending Beta**. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; crash gates
  report the artifacts present at the injected point.
- **clean control:** a normal append publishes, chains, and leaves the tree clean.
- **fault-injection control:** §11's mutant list, four-part kill rule — prove each red comes **from
  its injected defect**.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** no fleet dependency — must pass with all rigs down. Any
  `UNAVAILABLE` arm is a finding.
- **audit claim scope:** §0's dataset facts are **measured on VM101 this session**. The scraper
  characterization is from **ser8 Revision 1.5** — **not** a repo file, because none existed until
  it was tracked.
- **searched surfaces:** tracked repo; ser8 `~/Downloads/` and `~/cluster_controller/` (via
  Michael); the live `daily3.json` on VM101.
- **unavailable surfaces:** whether a **divergent** copy exists on VM101/Zeus — §7's stop condition
  closes it; the remote page's row ordering, which **no gate may depend on**.
