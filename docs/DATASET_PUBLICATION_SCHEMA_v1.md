# DATASET_PUBLICATION_SCHEMA_v1.md — the frozen dataset publication schema

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_IMPLEMENTATION.md` (REV1) §1.1–§1.5.
**Evidence:** `docs/PHASE_6_P0_SCOPING_v1.md` §4, §5, §6, §8.
**Status:** FROZEN as of Phase 6-P0. `manifest_schema_version: 1`.
**Scope:** the **combined** `daily3.json` dataset only. The split files
`daily3_midday.json` / `daily3_evening.json` are **not** covered — they remain unversioned and
unbound (scoping §8.3, an open item, deliberately not addressed here).

This document is frozen **before** the first version file is written, because the first version
file's name is permanent (instructions §1.1).

---

## 1. Layout — publish in place

Publication happens **in the repository root**, beside the existing dataset. This was decided on
evidence in scoping §4 and is not revisited here: no hardcoded literal sits on the certifying
Step-1 → miner → NPZ path, every link from `--lottery-file` (`window_optimizer.py:1258`) to
`open(path)` (`miner/range_miner_worker.py:565`) is a parameter, and moving the authority would
break **17 call sites in 13 files** while publishing in place breaks **zero**.

```
~/distributed_prng_analysis/
  daily3-<UTC>Z-<sha256[:12]>.json   immutable version files (append-only, never rewritten)
  daily3_current.json                the pointer manifest, atomically replaced
  daily3.json                        UNCHANGED — the pointer's resolved target (the alias)
```

`daily3.json` is not moved, renamed or modified. It is the compatibility alias every existing
consumer already opens.

> **Naming caution.** `daily3_current.json` is a *pointer manifest*, not a dataset. It sits
> beside `daily3_midday.json` and `daily3_evening.json`, which **are** datasets (session splits).
> The `_current` suffix names a different kind of object than the `_midday` / `_evening` suffixes
> do. The name follows scoping §4.3's recommended layout; the ambiguity is recorded rather than
> silently accepted.

## 2. The version filename grammar

```
daily3-<UTC>Z-<sha256[:12]>.json
       └──┬──┘  └─────┬─────┘ └─┬─┘
          │           │         └── load-bearing, see §5
          │           └──────────── content identity: first 12 hex of the file's sha256
          └──────────────────────── UTC instant, ISO-8601 basic: YYYYMMDDThhmmssffffff
```

`version_id` is the filename **without** the `.json` extension, so `filename == version_id + ".json"`
always holds and neither can drift from the other.

**Why both a timestamp and a digest.** Team Beta's ruling (project facts §2.10) requires version
IDs to carry a UTC timestamp **and** content identity. Either alone is insufficient: two
publications could in principle carry the same content (the timestamp separates them and records
ordering), and a timestamp alone would let two different byte-streams share a name under clock
skew or a re-run. The digest prefix also makes the filename self-verifying at a glance — a reader
can `sha256sum` the file and compare the first 12 characters without opening the manifest.

**Why 12 hex characters.** 48 bits. Enough that accidental collision within a lineage is not a
practical concern, short enough that the filename stays readable. The manifest always carries the
**full** 64-character digest; the prefix in the filename is a convenience, never the authority.

## 3. The pointer manifest — `daily3_current.json`

Written temp-then-`os.replace`, so a concurrent reader observes either the whole old manifest or
the whole new one, never a partial write. `os.replace` is atomic within a filesystem.

This is a **manifest**, not a bare symlink — Beta's words, recorded in project facts §2.10.

| Field | Type | Reasoning |
|---|---|---|
| `manifest_schema_version` | int | The one field **added** beyond the instruction §1.3 list. A frozen schema that carries no version marker cannot be evolved safely: a future reader has no way to tell schema 1 from schema 2 except by guessing at key presence. Flagged explicitly as an addition. |
| `version_id` | str | The immutable identity of this dataset version. Equals `filename` minus `.json`. |
| `filename` | str | The version file to resolve, **relative** to the manifest's own directory. Deliberately not absolute: the manifest must stay valid when the tree is cloned or the repo moves. |
| `sha256` | str | Full 64-hex digest of the version file. **The authority for content identity.** The filename prefix is a convenience copy of the first 12 characters. |
| `size_bytes` | int | Cheap first-line integrity check; catches truncation before a full re-hash is spent. |
| `record_count` | int | The semantic count. Distinguishes "same size, different content" and is what the P2 publication-prefix wall (a **record-sequence** check, never a byte-prefix test — project facts §2.10) will compare against. |
| `first_draw` | `{date, session}` | Lineage span, lower bound. Lets a consumer reject a version that does not cover the window it needs, without parsing 18k records. |
| `last_draw` | `{date, session}` | Lineage span, upper bound. The field a scraper advances on append. |
| `published_utc` | str, ISO-8601 `…Z` | When this version was published, distinct from when the data was scraped. Redundant with the timestamp inside `version_id` by construction, and carried separately so a consumer never has to parse the filename to learn it. |
| `dataset_lineage_id` | str | Which lineage this version belongs to. A correction opens a **new** lineage (§4). |
| `predecessor_sha256` | str \| null | The digest of the version this one supersedes **within the same lineage**; `null` for a lineage's first version. Makes the lineage a verifiable chain rather than a set of files sharing a prefix. |
| `notes` | object | Free-form provenance a future reader needs so a settled question is not re-litigated. For version one this records the 22 zero-draw records (§6). |

**No sidecar files.** The digest lives *inside* this manifest. A `.sha256`, `.txt` or `README.md`
companion is **not** matched by `.gitignore:41` `*.json`, would appear as an untracked file, and
would therefore make `repository_tree_clean` False and block certification at
`utils/run_finalizer.py:1589` (scoping §5.1–§5.2).

**The manifest must not be named `*_config.json` or `schema_*.json`.** Those two `.gitignore`
negations (`:43`, `:44`) are live and would un-ignore it, defeating §5.

## 4. Lineage and the correction protocol — documentation only in P0

**No enforcement code exists in P0.** This section records the contract that P2 (the scraper)
will implement.

- `dataset_lineage_id` for the bootstrap lineage is **`daily3-combined-L001`**. Corrections
  increment: `L002`, `L003`, ….
- **Normal append.** A new draw produces a new version file **in the same lineage**, with
  `predecessor_sha256` set to the previous version's digest. History is never rewritten.
- **Correction.** If a previously published record is found to be wrong, the scraper **does not
  rewrite it**. It halts with `CORRECTION_REQUIRED` and **may not create the corrected lineage
  autonomously** (project facts §2.10). A human opens a **new lineage** (`L002`), whose first
  version has `predecessor_sha256: null` — corrections break the chain by design, because the
  corrected data is not a continuation of the wrong data.
- **The old lineage is preserved and audit-retained.** It is never deleted. Certified generations
  produced against it stay reproducible against their own inputs, which is the whole point:
  `dataset_sha256` is a mandatory, verified field on every stripe assignment
  (`miner/range_miner_worker.py:640-652`), and the 6-P1 accumulator wall is an **exact
  input-manifest digest match**.
- **Append-only does not make prior scores valid on the next version.** Adding a draw changes
  windows, eligibility, gap/skip features, global frequency, normalization and any "latest N"
  (project facts §2.10). Version chaining records lineage; it does not license reuse of results
  across versions.

## 5. The `.json` extension is load-bearing

`.gitignore:41` is `*.json`. That single line is what keeps every published artifact invisible to
`git status --porcelain`, and therefore to `repository_tree_clean`
(`window_optimizer_integration_final.py:115-118`) and the certification wall at
`utils/run_finalizer.py:1589-1596`.

A version file with **any other extension** — or any non-`.json` sidecar beside it — would dirty
the working tree and **block certification**. This is not a style preference; it is the
constraint that makes publish-in-place viable at all.

Note that **gate 22** (`tests/test_s172_phase4_coordinator.py:1607-1610`) is *not* this wall: it
filters `endswith(".py")` and would never see a published `.json`. The finalizer is the wall that
matters.

## 6. Version one — the 22 zero-draw records

The bootstrap version contains **22 records with `draw == 0`**. They are **genuine data** and are
published unmodified. Recorded here so the question is not re-opened (scoping §6.2):

1. **The records are real draws.** `000` is a legitimate California Daily 3 outcome. Across
   18,068 draws the expected count of any single value is 18.07; the observed count of `0` is 22,
   within the observed per-value range of 4–32 (median 18). They are spread across the whole span
   — indices 254 (`2000-09-11 evening`) through 16,939 (`2024-08-11 evening`) — not clustered at
   an import boundary the way a placeholder artifact would be.
2. **The defect is in two consumers, and both are off the certifying path.** Both use the
   falsy-zero idiom `entry.get("draw") or …`: `digit_sequential_sieve.py:161-162` and
   `coordinator.py:1881`. The certifying loader is `load_residue_window`, which uses
   `entry.get("full_state", entry["draw"])` (`miner/range_miner_worker.py:575`) — a plain
   subscript, **zero-safe**.
3. **`DAILY3_CONSUMER_CONTRACT_v1.md` §9 MUST #5** states normatively that `0` is valid and must
   be emitted as `0`. A producer that dropped or rewrote these records would violate the frozen
   spec to accommodate two consumers already documented as defective.

The file is published as-is **because the data is correct** — not because changing it would be
inconvenient. Reproducibility is the reason not to "fix" it later, once someone notices the 22
records again; it is not the primary reason.

Fixing the two droppers is a separate, non-certifying change and **must never** be bundled with a
data publication.

## 7. What P0 deliberately does not do

Deferred to **P0.5** (changes to running code — mixing them with file creation would make the
first certification after publication ambiguous):

- Pointer resolution at WATCHER's param merge (`agents/watcher_agent.py:1385`).
- Absolutising the dispatched dataset path.
- Freeze-at-run-start: moving `dataset_sha256` from per-trial derivation at
  `miner/range_miner_coordinator.py:3499` to a run-level input.
- Classifying `range_miner_worker.py`'s bare `FileNotFoundError` (`:530-532`) as a `ResidueError`.
- The fleet provisioning manifest and fail-before-dispatch enforcement. Requires rigs.

Deferred to **P2** (the scraper): append-only production of new versions, the publication-prefix
wall, `CORRECTION_REQUIRED` halt behaviour, re-enabling `daily3scraper.service`, and any binding
between the combined file and the two session splits.

**Nothing reads this manifest yet.** No consumer resolves the pointer; that wiring is P0.5. P0
publishes the artifact and the verifier that proves it is internally consistent.
