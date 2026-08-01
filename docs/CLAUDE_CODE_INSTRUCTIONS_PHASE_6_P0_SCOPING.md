# CLAUDE_CODE_INSTRUCTIONS_PHASE_6_P0_SCOPING.md — REV1

**Phase 6-P0 scoping: where does a published dataset live, and what would break when it moves?**

**READ-ONLY SCOPING. Do not create directories, publish anything, move any file, or change any
code or configuration. Do not commit.** The deliverable is a report that lets the P0
implementation brief be written with real paths and a known blast radius.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do NOT
commit, push, or run WATCHER. STOP at the gate.

**The rigs are powered off tonight.** Anything requiring them is `UNAVAILABLE` (VIR-5), not
assumed. Do not report rig-side facts you could not observe.

---

## 0. Why this brief exists

Team Beta ruled the **form** of dataset publication — immutable versioned files, an atomic
pointer manifest (not a bare symlink), version IDs carrying UTC timestamp **and** content
identity, freeze-at-run-start, fail-before-dispatch, and a correction protocol that requires a
new lineage. **No location was ever specified.** A repo-wide search for
`source_location|destination_path|datasets/|dataset_root|publication_dir` returns exactly one
hit, and it refers to the fields abstractly.

Current observed state on VM 101:

```
/home/michael/distributed_prng_analysis/daily3.json   1,380,711 bytes, mtime 2026-03-04 16:58
/home/michael/datasets                                does not exist
```

So this is a blank slate. Alpha's working proposal, **to be tested by this scoping, not
assumed**:

```
~/datasets/daily3/daily3-<UTC>-<sha256prefix>.json   immutable, never modified
~/datasets/daily3/current.json                        pointer manifest, atomic replace
```

with rigs continuing to read `~/distributed_prng_analysis/daily3.json`, provisioned from the
published version and digest-verified on the target node.

**The question this brief must answer is not "is that a nice layout" — it is "what breaks."**

## 1. The falsifiable question

> If the authoritative dataset moves to a published, versioned location, **which consumers
> break, and what is the minimum change that keeps them working?**

## 2. Required findings

### 2.1 Every consumer of the dataset path
`docs/DAILY3_CONSUMER_CONTRACT_v1.md` enumerated consumers of the dataset's **content**. This
asks a different question: **who resolves the path?**

For every module, script, manifest, config or test that references `daily3.json` (or a
sibling — `daily3_midday.json`, `daily3_evening.json`, `lottery_history.json`,
`pa_pick3.json`): how is the path obtained? Hardcoded literal · CLI argument · config key ·
manifest entry · environment variable · relative to CWD · relative to `__file__`? **Give
`file:line` for each.**

**Then classify:** which consumers could read a *published* path with no code change (because
the path is already a parameter), and which have it baked in.

### 2.2 The relative-path hazard
Which consumers resolve `daily3.json` **relative to the current working directory**? Those
break silently if the file moves or if the process starts elsewhere — and a silent break here
means a run against a missing or wrong dataset. This is the highest-consequence item in the
report.

### 2.3 What the rigs actually read
The rig workers read a local copy. **Establish from the repository and configuration what path
they expect** — `distributed_config.json`, the coordinator's payload construction, the miner
worker, agent manifests. **Do not SSH; the rigs are off.** Report this as a
repository-derived expectation, explicitly **unverified against a live rig** (VIR-5/VIR-6).

### 2.4 Would the proposed layout require moving the file at all?
A cheaper option exists and should be evaluated honestly: **publish alongside** the current
location rather than replacing it — i.e. `daily3.json` stays exactly where it is and becomes
the pointer's *target* for version one, with future versions published as siblings. Compare:

- **(a) move** to `~/datasets/daily3/` and repoint consumers;
- **(b) publish in place** — versioned files beside the existing path, `daily3.json` retained
  as either a copy of the current version or the pointer's resolved target.

**State which requires fewer changes and which is less likely to break something silently.**
Alpha has no preference and is not proposing (a) — it was a first sketch.

### 2.5 The `.gitignore` interaction
`daily3.json` is gitignored (`.gitignore:41: *.json` per the TRSE audit). Would published
versions land inside or outside the repository, and does either choice interact with gate 22,
the clean-tree checks used by `finalize_run`, or the `repository_tree_clean` provenance field?
**A published dataset that makes the tree dirty would block certification** — check this
specifically.

### 2.6 Bootstrap: is the current file publishable as-is?
Version one would be the existing `daily3.json`. Establish:
- its sha256 and record count;
- whether it parses cleanly and satisfies the schema in
  `DAILY3_CONSUMER_CONTRACT_v1.md` §9;
- whether anything **blocks** publishing it unchanged.

**Note:** the consumer contract found 22 records where `draw == 0` that two loaders silently
drop. **Alpha's position is that this is a consumer defect, not a data defect, and the file
should be published as-is** — altering it would make the D6 certified generation
(`gen-20260730T002104136270Z`, commit `b08c2c5`) unreproducible against its own inputs.
**Confirm or challenge that reasoning**; do not change the file either way.

### 2.7 The pointer manifest's readers
Beta requires a pointer manifest, atomically replaced. **What would read it?** Nothing does
today. Identify where pointer resolution would have to be inserted so that a run resolves the
pointer **once at start** and every node uses that frozen version — and say whether that is one
insertion point or several.

### 2.8 Minimum viable P0
Given the above, **what is the smallest change that satisfies Beta's ruling?** Beta stated P0
needs *"one valid immutable version and a pointer manifest"*, produced by a manual bootstrap.
Enumerate concretely what must exist, and what can honestly be deferred to P0.5 (fleet
provisioning) and P2 (the scraper).

## 3. Out of scope

- **Do not create, move, copy or publish anything.** No directories, no version files, no
  pointer manifest.
- Do not modify `daily3.json`, `.gitignore`, any config, manifest or code.
- Do not SSH to the rigs — they are off.
- Do not implement pointer resolution, provisioning or verification.
- Do not re-derive the dataset's **content** schema — cite `DAILY3_CONSUMER_CONTRACT_v1.md`.
- Do not design the scraper — that is 6-P2.

## 4. Verification-integrity controls (VIR-1…6)

- **execution proof** — every consumer claim carries a `file:line` read this session. Computing
  the file's sha256 and record count is permitted (read-only).
- **clean control (VIR-2)** — state which consumers you verified as **already path-parameterised
  and safe**. A report listing only breakage gives no evidence the rest was checked.
- **fault-injection control** — n/a for read-only scoping; **say so** rather than omitting it.
- **completion sentinel (VIR-3)** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` plus a
  coverage table.
- **unavailable-observer (VIR-5)** — the rigs are off; rig-side facts are `UNAVAILABLE`, and
  repository-derived expectations must be labelled as such.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces. **The repository is
  not the system** — host paths, systemd units and deployed copies are separate surfaces, and
  the rig copies were not observed.

## 5. Deliverable

`docs/PHASE_6_P0_SCOPING_v1.md`:

1. **Path-resolution inventory** — every consumer, how it obtains the path, `file:line`,
   classified parameterised vs hardcoded.
2. **Relative-path hazards** — the silent-break list.
3. **Rig-side expectation** — repository-derived, marked unverified.
4. **(a) move vs (b) publish-in-place** — which is smaller and safer, with reasoning.
5. **`.gitignore` / clean-tree interaction**, including whether publication could dirty the tree
   and block certification.
6. **Bootstrap readiness** — sha256, record count, schema conformance, and a confirm-or-challenge
   on publishing as-is.
7. **Pointer-resolution insertion points** — one or several.
8. **Minimum viable P0**, with explicit P0.5 / P2 deferrals.
9. **Coverage table + completion sentinel.**

Then STOP for Team Alpha review.
