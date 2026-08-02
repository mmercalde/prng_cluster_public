# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_P2_SCRAPER.md — REV2 (DRAFT — pending Beta)

**S172 — Phase 6-P2: append-only immutable dataset publication.**

**Base:** HEAD (current). Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.

**Authority:** `docs/DATASET_PUBLICATION_SCHEMA_v1.md` (FROZEN, `manifest_schema_version: 1`).
**Where this brief and the schema differ, the schema wins** — it was frozen before the first
version file was written, deliberately, because that filename is permanent.

---

## REV1 → REV2

REV1 was a two-phase locate-then-scope brief because `daily3_scraper.py` is **not in the
repository** — never in git history, not gitignored. **It has now been located and read**, so REV2
is a single implementable document. REV1's Phase A survives only as §1's short on-host
confirmation.

**Located:** `ser8:~/Downloads/daily3_scraper.py` and `ser8:~/cluster_controller/daily3_scraper.py`
— **byte-identical**, both **Revision 1.5**, 4,334 bytes. That is the revision
`pa_pick3_scraper.py`'s header names as its parent, so the lineage is confirmed. A Syncthing
`.stversions` history exists (newest `20250810-165229`, 4,145 bytes) and is **superseded** — do not
work from it.

---

## 0. ⚠ READ FIRST — the scraper currently destroys the dataset

`main()` scrapes into `all_draws` and then:

```python
Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))   # OUTPUT_FILE = "daily3.json"
```

**It never reads the existing `daily3.json`.** It writes *only what it just scraped*, over the top.

And `--recent` does:

```python
if recent:
    start_year = end_year = TODAY.year
```

**Therefore `python3 daily3_scraper.py --json --recent` replaces the entire canonical dataset with
the current year alone.** That is the invocation a daily service would naturally use.

**This reframes a known item.** `daily3scraper.service` has been enabled since Sep 2025 with
`Restart=always`, targeting `run_daily3scraper.py` — **which never existed** — looping ENOENT every
boot. **That loop was protecting the dataset.** Had anyone "repaired" the unit by pointing it at
the real scraper with `--recent`, the 18,068-draw canonical dataset would have been truncated on
the next boot, silently, with no error.

**Consequences for this work:**
- **The service stays disabled until 6-P2 certifies**, and is never re-enabled against an
  unverified target.
- **Nothing in this brief may run the scraper against the real output path.** Every test writes to
  a temp directory.
- Fixing this destructive path is not a side effect of 6-P2 — **it is one of its primary
  purposes.**

## 0.1 The rest of the current behaviour, verified at source

| property | actual |
|---|---|
| write mode | **full overwrite**, `Path.write_text` — **not** atomic, no temp-then-replace |
| reads prior data | **no** |
| dedup | **none** — the PA descendant's header claims "deduped on (date, session, draw)"; the CA original does not dedup |
| sort | **none** — records are in scrape order, grouped by year then session; within-year order is whatever the page returns (**not verified**) |
| versions / lineage / manifest | **none** |
| correction detection | **none** |
| emitted fields | `date` (`YYYY-MM-DD`), `session`, `draw` (int) |
| `draw` parsing | `int(draw_str)` — so `"000"` → `0`, **correct** per the consumer contract |

**A crash mid-`write_text` truncates `daily3.json`.** There is no atomic boundary anywhere in the
current publication path.

---

## 1. On-host confirmation (short, read-only, do this first)

1. Does a copy of `daily3_scraper.py` exist on **VM101 or Zeus**? Report path and whether it is
   byte-identical to the ser8 Revision 1.5 (`sha256sum`). **If a divergent copy exists, STOP and
   report** — that is a second lineage and changes the target.
2. Read `daily3scraper.service` and report its **actual** `ExecStart`, `WorkingDirectory`, `User`
   and current enabled/active state.
3. Census beside the repo root, read-only: which of `daily3.json`, `daily3_current.json`,
   `daily3-<UTC>Z-<sha>.json`, `daily3_midday.json`, `daily3_evening.json` exist; sizes, record
   counts, sha256; and whether `daily3_current.json`'s recorded digest **matches** the version file
   it names. **Do not repair a mismatch — report it.**

---

## 2. The scope of the change

**Keep:** `fetch_draws`, the parsing, the row filtering, the CLI, the year loop. That logic is not
the defect and is not being revised.

**Replace entirely:** the publication path — everything from `all_draws` to disk.

**Structure it as a separate publication module** the scraper calls, so the publication contract is
independently testable and does not have to be exercised through the network. State the module and
the seam in the report.

## 3. What publication must do

Per schema §1-§3:

```
daily3-<UTC>Z-<sha256[:12]>.json   immutable version files, append-only, NEVER rewritten
daily3_current.json                the pointer manifest, temp-then-os.replace
daily3.json                        UNCHANGED — see §3.4, an OPEN QUESTION
```

### 3.1 Rules that are not negotiable

- **`version_id` is the filename minus `.json`**, so `filename == version_id + ".json"` always
  holds; neither can drift.
- Filename grammar: `daily3-<UTC>Z-<sha256[:12]>.json`, UTC ISO-8601 basic
  `YYYYMMDDThhmmssffffff`. The manifest carries the **full 64-hex** digest; the 12-hex prefix is a
  convenience, **never the authority**.
- **Manifest written temp-then-`os.replace`** — a concurrent reader sees the whole old manifest or
  the whole new one, never a partial write.
- **The `.json` extension is load-bearing.** `.gitignore:41` (`*.json`) keeps published artifacts
  out of `git status --porcelain`, hence out of `repository_tree_clean` and the certification wall
  at `utils/run_finalizer.py:1589`. **No sidecars** — not `.sha256`, not `.txt`, not `README`. The
  digest lives inside the manifest.
- **Never name anything `*_config.json` or `schema_*.json`** — `.gitignore:43,44` un-ignore those
  and defeat the arrangement.
- Manifest fields exactly per schema §3, including `record_count`, `first_draw`, `last_draw`,
  `dataset_lineage_id`, `predecessor_sha256`, `notes`.

### 3.2 The two walls

**Publication prefix — a RECORD-SEQUENCE check.** The previously published record sequence must be
a prefix of the new one. **A byte-prefix test is invalid for JSON arrays** — indentation, key order
or reformatting change bytes without changing records. Compare records.

**Accumulator input — an exact input-manifest digest match.** Not "compatible", not "prefix of".

**Append-only does NOT make prior scores valid on the next version.** One added draw changes
windows, eligibility, gap/skip features, global frequency, normalization and any "latest N".
Chaining records lineage; it does not license reuse of results. **Prefix-only merging is not
approved.**

### 3.3 The correction protocol — the hard rule

If a previously published record's value has changed:

- **do NOT rewrite it;**
- **halt with `CORRECTION_REQUIRED`;**
- **do NOT create the corrected lineage autonomously.**

A human opens `daily3-combined-L002`, whose first version has `predecessor_sha256: null` —
**corrections break the chain by design**, because corrected data is not a continuation of wrong
data. **The old lineage is preserved and audit-retained, never deleted**, so generations certified
against it stay reproducible.

Bootstrap lineage: **`daily3-combined-L001`**.

**Note the ordering problem this creates and solve it explicitly:** the current scraper produces
**unsorted, undeduped** records. A prefix check needs a **deterministic canonical record order**
before it means anything. Define that order, apply it to both sides, and state it in the report.
**Deriving the order from the scrape is not acceptable** — scrape order depends on the remote
page.

### 3.4 OPEN QUESTION for Beta — what happens to `daily3.json`?

Schema §1 says `daily3.json` is *"not moved, renamed or modified… the compatibility alias every
existing consumer already opens"* and simultaneously calls it *"the pointer's resolved target."*

**If the scraper stops writing it, it goes stale**; pointer resolution at WATCHER's param merge is
P0.5 work, and schema §7 says *"nothing reads this manifest yet."* **Alpha will not guess.**
Options: leave `daily3.json` frozen as the bootstrap alias and rely on pointer resolution; or have
publication also refresh it atomically as a convenience copy.

**Alpha's recommendation: leave it frozen and do not write it.** A scraper that still writes the
alias retains a second, unversioned publication path — exactly what 6-P2 exists to remove.
**Flagged for ruling; do not implement either option until Beta answers.**

## 4. Do not "fix" the 22 zero-draw records

Version one contains 22 records with `draw == 0`. **They are genuine data** — `000` is a legitimate
outcome; 22 sits inside the observed per-value range of 4-32 (median 18, expected 18.07); they span
indices 254 to 16,939 rather than clustering at an import boundary.

**The defect is in two consumers, both off the certifying path**, both using the falsy-zero idiom
`entry.get("draw") or …`: `digit_sequential_sieve.py:161-162` and `coordinator.py:1881`. The
certifying loader uses a plain subscript and is zero-safe (`miner/range_miner_worker.py:575`).

`DAILY3_CONSUMER_CONTRACT_v1.md` §9 MUST #5 states normatively that `0` is valid and must be
emitted as `0`. **A producer that dropped or rewrote these records would violate the frozen spec to
accommodate two consumers already documented as defective.** Fixing the droppers is separate,
non-certifying, and **must never be bundled with a data publication.**

## 5. Out of scope

The session splits `daily3_midday.json` / `daily3_evening.json` are **not covered by the frozen
schema** — unversioned and unbound, a deliberate open item. Any binding between combined and splits
is separate work. Also out: fixing the two zero-droppers; anything on the certifying Step-1 path;
re-enabling the service (that follows certification).

## 6. Gates — `tests/test_s172_phase_6_p2_publication.py`

**Every gate writes to a temp directory. No gate touches the real `daily3.json`, and no gate hits
the network** — feed the publication module fixture records directly.

- **G-DESTRUCTIVE-PATH-GONE:** the `--recent` truncation is impossible. Publish a full dataset,
  then publish a current-year-only scrape; **the published version still contains the full
  history**, and no path overwrites a dataset with a subset.
- **G-APPEND-ONLY:** a new draw yields a **new** version file; **no prior version file is modified
  or deleted** (compare digests of every prior file before and after).
- **G-PREFIX-WALL:** the prior record sequence is a prefix of the new one. **Include a reformatting
  case** — same records, different indentation/key order — which must **PASS**, proving the check
  is record-based and not byte-based.
- **G-CANONICAL-ORDER:** the same record set in different scrape orders produces the **identical**
  canonical sequence and digest.
- **G-CORRECTION-HALT:** a changed prior record halts with `CORRECTION_REQUIRED`; **no file is
  written; no `L002` is created.**
- **G-MANIFEST-ATOMIC:** manifest is temp-then-`os.replace`, same filesystem; a crash between write
  and replace leaves the old manifest wholly intact.
- **G-IDENTITY:** `filename == version_id + ".json"`; the 12-hex prefix matches the full digest's
  first 12; `record_count`, `size_bytes`, `first_draw`, `last_draw` all correct.
- **G-CHAIN:** `predecessor_sha256` links versions within a lineage; `null` on a lineage's first.
- **G-TREE-CLEAN:** after publication `git status --porcelain` is **clean** — proving the `.json`
  extension rule holds and **no sidecar was written**.
- **G-ZERO-DRAWS:** `draw == 0` records survive publication unmodified and uncounted-out.

**Mutants:** restore the overwrite-with-subset path · use a byte-prefix test (must red on the
reformatting case) · derive canonical order from scrape order · rewrite a prior record instead of
halting · create `L002` autonomously · write a `.sha256` sidecar · non-atomic manifest write · drop
`predecessor_sha256` · drop a zero-draw record.

## 7. Report

`docs/S172_PHASE_6_P2_IMPLEMENTATION_REPORT.md` — §1's three confirmations; the publication module
and its seam; **the canonical record order chosen and why it is not scrape-derived**; the prefix
wall's record-comparison evidence including the reformatting case; the correction-halt evidence;
gate/mutant counts; confirmation **no gate touched the real dataset or the network**; and the §3.4
question left unimplemented pending Beta. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; the
  append-only gate reports prior-file digests before and after.
- **clean control:** a normal append publishes successfully and leaves the tree clean.
- **fault-injection control:** §6's mutant list, four-part kill rule on each — prove each red comes
  **from its injected defect**.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** no fleet dependency; must pass with all rigs down. Any
  `UNAVAILABLE` arm is a finding.
- **audit claim scope:** the scraper characterization in §0/§0.1 is from **`ser8:~/Downloads/daily3_scraper.py`
  Revision 1.5**, read this session — **not** from a repo file, because none exists.
- **searched surfaces:** tracked repo; ser8 `~/Downloads/` and `~/cluster_controller/` (via
  Michael).
- **unavailable surfaces:** VM101 and Zeus host state — §1 closes this; the remote page's row
  ordering, which only a live fetch would establish and which **no gate may depend on**.
