# PHASE_6_P0_IMPLEMENTATION_v1.md — dataset version one published in place

**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_IMPLEMENTATION.md` (REV1).
**Evidence:** `docs/PHASE_6_P0_SCOPING_v1.md`.
**Base:** VM 101, `~/distributed_prng_analysis`, `main` @ **`32311c8`**, venv `~/venvs/torch`.
**Date:** 2026-08-01.

> Session-start `git rev-parse` returned `216c74b`, the base the instructions name. `32311c8` is
> Michael's own commit at 07:50:15 landing the implementation brief itself, made after that check.
> The delta between the two is **that one documentation file** — no code, no config, no manifest —
> so every `file:line` anchor in this report is identical under both. Nothing from this session is
> committed.

**P0 created files. P0 changed no running code.** No `.py` on the certifying path was modified.
`daily3.json` is byte-unchanged. Nothing consumes the manifest yet. **No commit, no push, no
WATCHER run, no SSH, no GPU work.**

---

## 0. Result up front

| Deliverable | Status |
|---|---|
| §1.1 publication schema, frozen before writing | ✅ `docs/DATASET_PUBLICATION_SCHEMA_v1.md` |
| §1.2 immutable version file, byte-exact | ✅ digest re-derived from disk, matches source |
| §1.3 pointer manifest, atomic, no sidecars | ✅ temp → `os.replace`, digest inside the JSON |
| §1.4 read-only verifier | ✅ `scripts/verify_dataset_publication.py`, 20/20 clean |
| §1.5 correction protocol, documentation only | ✅ schema doc §4; no enforcement code |
| §2 tree stays clean | ✅ **both published artifacts git-ignored; zero porcelain delta** |
| §4 VIR-2 fault injection | ✅ **7/7 mutants detected + clean scratch control** |
| §5 non-regression | ✅ phase4 **63/63**, chapter1-P0 **12/12** |

## 1. The frozen schema and why each field exists

Full contract: **`docs/DATASET_PUBLICATION_SCHEMA_v1.md`**, frozen *before* the version file was
written because §1.1 makes the first filename permanent.

### 1.1 The filename grammar

```
daily3-<UTC>Z-<sha256[:12]>.json
```

`version_id` is the filename minus `.json`, so `filename == version_id + ".json"` always holds and
the two cannot drift. Timestamp format is ISO-8601 basic `YYYYMMDDThhmmssffffff`, matching the
existing generation-ID convention (`gen-20260730T002104136270Z-…`).

**Both halves are required.** Beta's ruling (project facts §2.10) is that a version ID carries a
UTC timestamp **and** content identity. A timestamp alone would let two different byte-streams
share a name under clock skew or a re-run; a digest alone records no ordering. 12 hex = 48 bits,
short enough to stay readable, and the manifest always carries the **full** 64-hex digest — the
filename prefix is a convenience, never the authority.

**The `.json` extension is load-bearing, not cosmetic.** `.gitignore:41` `*.json` is the single
line that keeps published artifacts invisible to `git status --porcelain`, hence to
`repository_tree_clean` (`window_optimizer_integration_final.py:115-118`) and the certification
wall at `utils/run_finalizer.py:1589-1596`. Any other extension would dirty the tree and block
certification. Proven empirically in §3.

### 1.2 The manifest fields

| Field | Reasoning |
|---|---|
| `manifest_schema_version` | **The one field added beyond the §1.3 list**, flagged deliberately. A frozen schema with no version marker cannot be evolved: a future reader could only distinguish schema 1 from 2 by guessing at key presence. |
| `version_id` | Immutable identity of this version. |
| `filename` | Resolved **relative to the manifest's own directory** — deliberately not absolute, so the manifest stays valid across a clone or a move. |
| `sha256` | Full 64-hex digest. **The authority for content identity.** |
| `size_bytes` | Cheap first-line integrity check; catches truncation before a full re-hash is spent. |
| `record_count` | The semantic count. Distinguishes "same size, different content", and is what P2's publication-prefix wall — a **record-sequence** check, never a byte-prefix test (project facts §2.10) — will compare against. |
| `first_draw` / `last_draw` | Lineage span. Lets a consumer reject a version that does not cover its window without parsing 18k records; `last_draw` is what a scraper advances on append. |
| `published_utc` | When published, distinct from when scraped. Redundant with the `version_id` timestamp by construction, carried separately so no consumer has to parse a filename to learn it. |
| `dataset_lineage_id` | Which lineage this belongs to. A correction opens a **new** lineage. |
| `predecessor_sha256` | Digest of the version this supersedes **within the same lineage**; `null` for version one. Makes the lineage a verifiable chain, not a set of files sharing a prefix. |
| `notes` | Provenance a future reader needs so a settled question is not re-litigated — here, the 22 zero-draw records and why they are genuine. |

**No sidecar files.** The digest lives *inside* the manifest. A `.sha256`, `.txt` or `README.md`
companion is not matched by `*.json`, would show as untracked, and would block certification
(scoping §5.2). The manifest is named `daily3_current.json` — **not** `*_config.json` or
`schema_*.json`, whose `.gitignore` negations (`:43`, `:44`) are live and would un-ignore it.

> Recorded caution: `daily3_current.json` sits beside `daily3_midday.json` / `daily3_evening.json`,
> which **are** datasets. The `_current` suffix names a different kind of object (a pointer
> manifest) than the `_midday`/`_evening` suffixes do. The name follows scoping §4.3's recommended
> layout; the ambiguity is recorded rather than silently accepted.

### 1.3 Correction protocol — documentation only, no enforcement code in P0

Lineage `daily3-combined-L001`. Normal append → new version in the same lineage with
`predecessor_sha256` set. A correction → the scraper halts with `CORRECTION_REQUIRED` and **may
not** create the corrected lineage autonomously; a human opens `L002`, whose first version has
`predecessor_sha256: null`, because corrected data is not a continuation of wrong data. The old
lineage is preserved and audit-retained. **Append-only does not make prior scores valid on the
next version** (project facts §2.10).

## 2. What was published

Publication is **in place** (scoping §4, decision not revisited): `daily3.json` stays exactly
where it is and is the pointer's resolved target.

| File | sha256 | Bytes |
|---|---|---|
| `daily3-20260801T145551443433Z-513648160d35.json` | `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6` | 1,380,711 |
| `daily3_current.json` (manifest) | `ef3275264a906b7064e9e50b87d41e1e7a0c02071716a5b083ca0d6e57f3250a` | — |
| `daily3.json` (alias, **untouched**) | `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6` | 1,380,711 |

Content: **18,068 records**, `2000-01-01 evening` → `2026-02-26 midday`. Every figure matches
scoping §6 exactly.

**Execution proof (VIR-1).** The version file's digest was **re-derived from disk after writing**,
not assumed from the copy operation, and compared to the source before the manifest was written.
The manifest was written temp-then-`os.replace` with `fsync` on both the file and the containing
directory, so a reader sees the whole old manifest or the whole new one, never a partial write.

**Published as-is because the data is correct.** The 22 `draw == 0` records are genuine: `000` is a
legitimate Daily 3 outcome, expected count of any value across 18,068 draws is 18.07 against an
observed 22 (per-value range 4–32, median 18), and they are spread across indices 254
(`2000-09-11 evening`) to 16,939 (`2024-08-11 evening`) rather than clustered at an import
boundary. The two droppers use the falsy-zero idiom (`digit_sequential_sieve.py:161-162`,
`coordinator.py:1881`) and are **off** the certifying path; the certifying loader
`load_residue_window` (`miner/range_miner_worker.py:575`) uses a plain subscript and is zero-safe.
Their indices and this reasoning are recorded in the manifest's `notes` so the question is not
re-litigated. Reproducibility is the reason not to "fix" it later — not the primary reason.

The bootstrap was run from the **scratchpad**, not the repo, so no publisher `.py` landed in the
tree. P2 (the scraper) owns ongoing publication; this was the manual bootstrap Beta asked for.

## 3. Tree cleanliness — proven, not asserted

### 3.1 `git check-ignore -v` on every created file

```
daily3-20260801T145551443433Z-513648160d35.json   IGNORED  <- .gitignore:41:*.json
daily3_current.json                               IGNORED  <- .gitignore:41:*.json
scripts/verify_dataset_publication.py             NOT ignored
docs/DATASET_PUBLICATION_SCHEMA_v1.md             NOT ignored
```

### 3.2 Before / after `git status --porcelain`

**Before (pre-existing, `216c74b`) — 5 entries:**

```
?? CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md
?? CLAUDE_CODE_BRIEF_S177_RESUBMISSION_v1.md
?? CLAUDE_CODE_BRIEF_S178_FOLLOWUP_v1.md
?? CLAUDE_CODE_BRIEF_WATCHER_KPI_VALIDATION_v1.md
?? tmp/
```

**After — 8 entries:**

```
 M tests/test_s172_phase4_coordinator.py          <- gate-22 allowlist registration (§4)
?? CLAUDE_CODE_BRIEF_S176_FOLLOWUP_v1.md          <- pre-existing
?? CLAUDE_CODE_BRIEF_S177_RESUBMISSION_v1.md      <- pre-existing
?? CLAUDE_CODE_BRIEF_S178_FOLLOWUP_v1.md          <- pre-existing
?? CLAUDE_CODE_BRIEF_WATCHER_KPI_VALIDATION_v1.md <- pre-existing
?? docs/DATASET_PUBLICATION_SCHEMA_v1.md          <- deliverable, to be committed
?? scripts/verify_dataset_publication.py          <- deliverable, to be committed
?? tmp/                                           <- pre-existing
```

### 3.3 Reading this honestly

**The publication itself contributed zero entries.** Both published artifacts are matched by
`.gitignore:41` and are invisible to `git status`, which is precisely what §1.1's `.json`
requirement buys.

The three new entries are **not** publication artifacts — they are the deliverable's own source
files, which the instructions mandate (§1.4 the verifier, §1.1 the schema documentation, §5 the
allowlist registration) and which Michael commits. §2's "no new untracked non-ignored files" is
therefore satisfied **for the publication**, which is what it governs; it cannot also mean the
brief's own mandated deliverables must not exist.

**The four `CLAUDE_CODE_BRIEF_S17*.md` files and `tmp/` were not touched, deleted or committed.**
They are pre-existing and not P0's to fix. A certifying run would fail at
`utils/run_finalizer.py:1592` today because of them — that was true before this session and is
unchanged by it.

## 4. Gate 22 — registered, with a positive control

Gate 22 (`tests/test_s172_phase4_coordinator.py:1607-1610`) asserts `changed_py <= allowed` over
`git status --porcelain` filtered to `.py`.

**Positive control first.** Run *before* touching the allowlist, the suite returned **62/63** with
exactly:

```
AssertionError: unexpected changed .py files: {'scripts/verify_dataset_publication.py'}
```

That proves the gate is live and non-vacuous for this file rather than assuming it. The verifier
was then registered with rationale per the established pattern, and the suite returned **63/63**.

`tests/test_s172_phase4_coordinator.py` is the **only** `.py` this deliverable modifies. It is a
test harness, already in its own allowlist, edited only to add the registration comment block and
one string — explicitly authorised by instructions §5. It is not on the certifying path.

## 5. The verifier — clean control and fault injection

`scripts/verify_dataset_publication.py`. Read-only by construction: it verifies and never repairs;
no write, rename or delete path exists in it. Path resolution is `__file__`-anchored, never
`os.getcwd()` (scoping §2.3's non-hazardous pattern). No runtime consumer — nothing in the
pipeline imports it, the same shape as the already-registered
`scripts/extract_search_bounds_snapshot.py`.

**20 checks:** manifest existence/parse/shape/required-fields/schema-version/digest-format
(M0–M6), filename-grammar and embedded-digest agreement (F1–F3), version-file
existence/size/**re-derived digest**/JSON/array/record-count/first/last (V0–V7), and alias
existence + **alias digest agreement** (A0–A1).

### 5.1 Clean control (VIR-2) — `PASS`

```
DATASET_PUBLICATION_VERIFIER — SENTINEL
  checks run:    20
  checks failed: 0
  TERMINAL STATE: PASS
```

### 5.2 Fault-injection control (VIR-2) — 7/7 detected

**Run against a scratch copy of the publication set. The published artifacts were never touched**
(re-verified 20/20 afterwards, §6).

| Case | Corruption | Expected | Flagged | Exit | Result |
|---|---|---|---|---|---|
| FI-0 | *(none — clean scratch copy)* | — | none | 0 | **PASS** control |
| FI-1 | one digit flipped mid-file; **size and JSON validity preserved** | `V2` | `V2` | 1 | DETECTED |
| FI-2 | final record truncated | `V1 V2 V5 V7` | `V1 V2 V5 V7` | 1 | DETECTED |
| FI-3 | alias diverges from published version | `A1` | `A1` | 1 | DETECTED |
| FI-4 | manifest digest tampered | `F3 V2 A1` | `F3 V2 A1` | 1 | DETECTED |
| FI-5 | version file missing | `V0` | `V0` | 1 | DETECTED |
| FI-6 | `record_count` wrong | `V5` | `V5` | 1 | DETECTED |
| FI-7 | required field missing | `M3` | `M3` | 1 | DETECTED |

**FI-1 is the load-bearing case.** A single flipped digit leaves the file the same size and still
valid JSON — only the re-derived digest can catch it, and only `V2` fired. That is the difference
between a verifier and a file-exists check.

**FI-0 is what makes the other seven meaningful.** A clean scratch copy still passes, so the seven
failures come from the injected corruption and not from being in a scratch directory. Without it,
every mutant "detection" could have been the harness failing for an unrelated reason — the vacuous
detector this control exists to rule out.

**FI-3 is the failure mode scoping §4.2 warned about**, made detectable for the first time: an
alias that has drifted from the published authority. Note what the verifier adds that the miner's
existing guard cannot — the coordinator derives the expected `dataset_sha256` from the same path it
dispatches (`range_miner_coordinator.py:3499` → `:3421`), so that guard proves *agreement*, not
*authority*. `A1` compares the alias against an **independently published** digest.

## 6. Inertness — no existing run behaviour changed, and how I know

- **No `.py` on the certifying path was modified.** The only `.py` touched is
  `tests/test_s172_phase4_coordinator.py` — a test harness, allowlist registration only (§4). The
  full set of `.py` entries in `git status` is exactly that file plus the new verifier. `miner/`,
  `window_optimizer.py`, `window_optimizer_integration_final.py`, `sieve_gpu_worker.py`,
  `prng_registry.py`, `agents/`, PWC and ZMQ are all untouched.
- **`daily3.json` is byte-unchanged.** sha256 re-derived after publication:
  `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6` — identical to the §1.2
  value. mtime still `2026-03-04 16:58:27 −0800`; not moved, not renamed, not rewritten.
- **Nothing consumes the manifest.** Scoping §7 established that no pointer reader exists; P0 adds
  none. The only reader of `daily3_current.json` is the hand-run verifier.
- **Published artifacts intact after fault injection** — verifier re-run at the end: 20/20, `PASS`.
- **Non-regression:** `tests/test_s172_phase4_coordinator.py` **63/63**;
  `tests/test_chapter1_p0_corrections.py` **12/12**.
- **`.gitignore` untouched**, including the dead `:42` negation, which still needs a governed
  decision because fixing it would un-ignore an unknown set of files.

## 7. Out-of-scope lines — held

None of the following was crossed. Each was reachable and was deliberately left alone.

- No pointer resolution at WATCHER, no path absolutising, no freeze-at-run-start, no
  `FileNotFoundError` → `ResidueError` classification. **All P0.5.**
- `daily3.json` not moved, renamed or modified.
- `.gitignore` not touched.
- **D1 falsy-zero droppers not fixed** (`digit_sequential_sieve.py:161-162`, `coordinator.py:1881`).
- **The §2.1 WATCHER check-one-path/use-another defect not fixed** — preflight resolves
  `<REPO_ROOT>/daily3.json` (`watcher_agent.py:489`) while dispatch is `Popen(cmd, …)` at `:1948`
  with no `cwd=` and the child receives the bare string. Real, reported, not P0. **Worth noting
  that publication has now created a second copy of the dataset in the tree, which is the
  condition scoping §2.1 identified as making this defect materially more likely to bite.** Both
  new files are `.json` version-stamped names, not `daily3.json`, so no CWD can now resolve
  `daily3.json` to a published version by accident — but the defect itself is unchanged and P0.5
  should take it.
- `daily3_midday.json` / `daily3_evening.json` not touched. P0 publishes the **combined file
  only**; the splits remain unversioned and unbound (open item).
- No SSH, no WATCHER, no sieve, no GPU kernel, no pipeline.

## 8. Deferrals restated, so the boundary is on the record

**P0.5 — changes to running code, and anything needing a rig:**
pointer resolution at WATCHER's param merge (`agents/watcher_agent.py:1385`); absolutising the
dispatched dataset path; freeze-at-run-start (moving `dataset_sha256` from per-trial derivation at
`miner/range_miner_coordinator.py:3499` to a run-level input); classifying the bare
`FileNotFoundError` at `range_miner_worker.py:530-532` as a `ResidueError`; the fleet provisioning
manifest and fail-before-dispatch enforcement.

**P2 — the scraper:** append-only production of new versions; the publication-prefix wall (a
record-sequence check, **never** a byte-prefix test); `CORRECTION_REQUIRED` halt behaviour;
re-enabling `daily3scraper.service`; regeneration and binding of the two session splits.

**Not P0, should be scheduled:** the D1 falsy-zero fix; the `.gitignore:42` dead negation; the
§2.1 WATCHER defect.

## 9. Verification-integrity controls (VIR-1…6)

- **execution proof:** the version file's digest was re-derived from disk after writing (§2), not
  inferred from the copy; gate 22's registration was preceded by a run proving it *fails* without
  it (§4); every test tally is a real suite run, not a claim.
- **clean control:** verifier 20/20 on the real publication (§5.1) **and** FI-0, a clean scratch
  copy that still passes (§5.2).
- **fault-injection control:** 7 mutants, each detected, each naming the expected check (§5.2).
  Explicitly **not** skipped — this is the control the instructions flag as easiest to omit.
- **detector independence:** the alias check (`A1`) compares against an independently published
  digest rather than re-deriving both sides from one path — the distinction the miner's
  coordinator-side guard does not make (§5.2).
- **completion sentinel:** §10.
- **unavailable-observer (VIR-5):** the three rigs are **powered off**. No SSH was attempted.
  Nothing rig-side was observed or assumed; no rig claim appears in this report. Nothing in P0
  required them.
- **audit claim scope (VIR-6):** **searched** — the live repo tree at `32311c8` on VM 101:
  `git status --porcelain` (before and after), `git check-ignore -v` on all four created files,
  `.gitignore:38-48`, gate 22's full allowlist and assertion
  (`tests/test_s172_phase4_coordinator.py:1602-2036`), a full parse of all 18,068 records of
  `daily3.json`, `sha256sum` and `hashlib` digests computed independently, and two full test
  suites executed. **Not searched / unavailable** — the three rig CT100s and bare-metal `.127`
  (powered off); host surfaces outside the repo (systemd, cron); the ~200 archival dataset copies;
  other git branches. **The repository is not the system.**

## 10. Completion sentinel

```
PHASE_6_P0_IMPLEMENTATION_v1 — SENTINEL

OVERALL:  PASS

  §1.1 schema frozen before writing ..... PASS  (docs/DATASET_PUBLICATION_SCHEMA_v1.md)
  §1.2 immutable version file ........... PASS  (byte-exact; digest re-derived from disk)
  §1.3 pointer manifest ................. PASS  (atomic os.replace; no sidecars; digest inside)
  §1.4 read-only verifier ............... PASS  (20/20 clean control)
  §1.5 correction protocol documented ... PASS  (documentation only; no enforcement code)
  §2   tree stays clean ................. PASS  (both artifacts git-ignored; zero delta)
  §4   fault-injection control .......... PASS  (7/7 mutants + FI-0 clean scratch control)
  §5   non-regression ................... PASS  (phase4 63/63; chapter1-P0 12/12)

  rig-side .............................. UNAVAILABLE (rigs powered off; nothing claimed,
                                                       nothing required by P0)

  PUBLISHED:
    daily3-20260801T145551443433Z-513648160d35.json  sha256 513648160d35…68f6  1,380,711 B
    daily3_current.json                              sha256 ef3275264a90…250a

  CODE CHANGED: none on the certifying path.
    tests/test_s172_phase4_coordinator.py — gate-22 allowlist registration ONLY (§5-authorised).
  daily3.json: BYTE-UNCHANGED (513648160d35…68f6), not moved, not renamed.
  Pre-existing dirty entries (4 briefs + tmp/): untouched, not deleted, not committed.

  NOT COMMITTED. NOT PUSHED. NO WATCHER RUN. NO SSH. NO GPU WORK.

  STOPPED AT THE GATE FOR TEAM ALPHA REVIEW.
```
