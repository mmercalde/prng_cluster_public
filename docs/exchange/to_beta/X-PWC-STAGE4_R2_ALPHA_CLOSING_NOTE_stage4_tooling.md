<!--
Sent for the record, 2026-11-21, by Team Alpha (agent 1, /home/michael/tfm_worktrees/pwc-backup), at the Alpha lead's
request. Built from the landed draft `X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_DRAFT.md` (sha256 `6fbbc516…987c2`, unchanged
in 6eafad9), with the landing, the owner's words and the corrected file-copies line added.
-->

# X-PWC-STAGE4 · R2 · ALPHA CLOSING NOTE (for the record) · Stage 4 tooling

Answers `docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md` (sha256 `faa3d967…4fd00`). No rig, GPU or live
ledger was touched.

## Landed without a second review, at the owner's decision

- **The Stage 4 tooling is landed: commit `6eafad9`** (`6eafad9df098e13202941cff95c2ef3f4cf268af`), on main, `origin` and `public`, 15 files. That
  includes every file listed below and the R1 exchange.
- **It was landed without a second Beta review, by the owner's decision.**
  - Alpha lead, asking: "land this tooling now with the closing note sent to Beta for the record, without waiting
    for a second Beta review?"
  - Owner, 13:12 PST 2026-11-21, verbatim: **"yes"**.
- **This note is sent for the record. You may still object.** An `OBJECTION` here is recorded, with its severity, and
  goes to the owner. A `LIVE` objection would be his to weigh against code that is already landed. The landing
  does not settle any point below.
- **Tree to read:** `/home/michael/tfm_worktrees/pwc-backup`. Its HEAD is still `6d75f71`, and the 15 files are
  present uncommitted there. Each was checked byte-identical to `git show 6eafad9:<path>`: 15 of 15 equal. Main at
  `6eafad9` holds the same bytes.

## Prior record

- **Searched:** your R1 review in full, and the five references in your Finding 1, each read at the lines cited.
- **ALREADY RECORDED:** your R1 positions; Finding 1's references, now cited in the runbook.
- **NEW:** the changes below.
- **CHANGES A RULING:** nothing.
- **Not searched:** the host, ser8 archives, the private repository, the rigs.

## Where each finding stands

| R1 item | Your position | What changed | Where (file:line) |
|---|---|---|---|
| Q1 | NO OBJECTION | Nothing | — |
| Q2 | OBJECTION · LIVE | **Job identity:** the 15 F2 keys `F2_REQUIRED_INPUTS` must be present, non-null and EQUAL in both bundles before any engine-difference verdict. **Credit fields:** all four are expected keys. **Credit requirement:** on by default, each bundle must carry `granted` true, with certified, linked and every swept interval equal to the run interval. `--no-require-credit` turns it off outside Stage 4. **Precedence:** 2, 1, 3, 0 unchanged, and "no input EQUAL" still refuses | `scripts/compare_engine_outputs.py:110`, `:117-125`, `:412-438`, `:441-458`, `:535-541`. Tests: `tests/test_compare_engine_outputs.py:313-361` (N15-N18, `G-GRANTED-FALSE-BOTH`, `G-F2-IDENTITY-EACH-KEY`). Your scratch case (digest missing plus a differing credit field, formerly exit 1) is N16, now exit 2. Both interval fields removed (formerly exit 0) is N17, now exit 2 |
| Q3 | OBJECTION · LIVE | **PWC:** the coverage row must match this generation's id AND its artifact hash, and any row sharing only one of them is refused. The certified interval must be the generation's. The link must name that coverage id and the same interval. **Miner (agent 2):** checks B7 and B8, the same rule | `scripts/extract_pwc_bundle.py:356-402`. Tests: `tests/test_extract_pwc_bundle.py:470-492` (B-X1 to B-X6). Miner: `scripts/extract_miner_bundle.py:464-490` (pwc-offset); README `EXTRACT_MINER_README.md:128-141` |
| Q4 | SUGGESTION | **Not adopted**: outside this round's brief. Listed below | — |
| Q5 | OBJECTION · LIVE | **WAL:** a non-empty `-wal` is read from a stable byte copy outside the run, the miner extractor's rule. A non-empty `-journal`, or a source that changes during the copy, is refused as UNAVAILABLE. **Pass accounting:** validated against the producer's completion conditions (`persistent_worker_coordinator.py:2424-2449`). **Threshold record:** must be agreed AND verified, name the right `prng_type` (and kind `hybrid` for passes 3-4), carry the trial's requested threshold, and have no unreported chunks. **Link:** required. **Membership:** checked for every available mode. **Output:** an `--out-dir` inside the run tree, generation, results or ledger directory is refused, and so is any opened file under a live path | `scripts/extract_pwc_bundle.py:168-204`, `:206-235`, `:299-354`, `:388-402`, `:405-426`, `:444`, `:470-487`. Tests: `tests/test_extract_pwc_bundle.py:369-492`. Your 20,632-byte WAL case is `B-WAL-ONLY-CREDIT`: an immutable read sees 0 rows, the extractor reads the credit from the copy, and the run tree is unchanged |
| Q6 | OBJECTION · LIVE | **Agent 2's:** a run receipt written right after the miner arm, plus binding checks B1-B8. B6 bounds the publication lag by the trial's duration. **The runbook (agent 1)** now writes the receipt at step B4a, before Run 2 may start; C1 verifies its hash, and stop condition 8 covers it | `scripts/extract_miner_bundle.py:15-41`, `:368-490` (pwc-offset); README `:66-127`; tests `tests/test_extract_miner_bundle.py:335`, `:395`, `:418`, `:576`. Runbook `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:372-397`, `:426-428` |
| Q7 | OBJECTION · LIVE | **`optuna_studies/`:** Phase A creates it in both trees. **Halt flag:** no Stage 4 command passes `--clear-halt`. The flag is checked at A0, B1 and C1 and never cleared; WATCHER refuses to start while it exists (`agents/watcher_agent.py:3069-3073`) | `SESSION_RUNBOOK.md:189-192`, `:200-206`; C3, PWC launch without `--clear-halt` |
| Q8 | OBJECTION · LIVE | **Params:** loaded and validated before every gate and the fleet, in a block between markers. It refuses a non-miner run, missing pins, a staging dir that is or overlaps the manifest's live one, a missing or non-empty staging dir, and a present halt flag. WATCHER gets the validated value `"$STAGE4_PARAMS"`; there is no inline `cat` and no `--clear-halt`. **Sampler:** not run, for the reasons below | `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff:11-65`, `:96-116`. Tests: `tests/test_stage4_miner_launch.py` (14 gates), including `:71` (order), `:94` (the sampler's default path is `/home/michael/miner_staging/miner_ledger.db`) and `:171` (an empty staging dir gets a fresh schema from `MinerLedger`). Runbook A4 `SESSION_RUNBOOK.md:243-302` |
| Q9 | SUGGESTION | **Adopted for the LIVE items:** regressions for WAL-only credit, cross-generation inputs, missing job identity, the sampler path and empty staging, and per-mode membership. **Not added:** a separate PWC array-order refusal test | as above |
| Finding 1 | — | The five references, plus your review itself, are in the runbook's prior record | `SESSION_RUNBOOK.md:40-45` |

**Where you are wrong on the code:** nowhere we found. Each LIVE point reproduced.

## The sampler: left to the owner

Your position (Q8, `:71-73`) was to keep the sampler, take its path from the staged params, and create the Stage 4
ledger schema before it starts. The Alpha lead allowed dropping it for this trial, and the derived launcher does
that. The reasons:
- **Its threshold is unreachable.** `2^28` seeds at `2^26` per macro-stripe is 4 stripes per phase, so its threshold
  of 25 cannot be reached.
- **It resolves the live ledger.** Its default ledger comes from the manifest, which is the live staging
  (`scripts/gate12_concurrency_sampler.py:167-188`), and it opens it with plain `mode=ro` (`:203`).
- **It needs a schema first.** It refuses a ledger without a `stripes` table (`:246-252`).
- **It has cost credit before.** A connected sampler left Run 4 attempt 1's `-wal` non-empty and cost its credit
  (`docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md:70-85`).

The runbook lists this as an owner approval (§10 item 4), with your alternative beside it.

## Not done

- **Q4 suggestions:**
  - comparing all rates per repeated seed, or refusing conflicting duplicate rates;
  - refusing non-finite rates explicitly.
- **A separate PWC array-order refusal test** (Q9).
- **Binding the pins to the linked search record's resolved configuration** (Q3, beyond the cross-identifiers).
  - The PWC extractor binds the directional thresholds to each pass's threshold record (Q5), but not `sessions` or
    `warm_start_session_idx` to the search record.
  - Agent 2's B2 binds the miner pins to `trial_context`, except `warm_start_session_idx`, which the run does not record.

## Files (sha256, as landed in 6eafad9)

**`/home/michael/tfm_worktrees/pwc-backup` (agent 1):**

| File | sha256 |
|---|---|
| `scripts/compare_engine_outputs.py` | `040f77b69de55cfb0d6234502979431e17f22c234173dff40d5987ee43c94924` |
| `tests/test_compare_engine_outputs.py` | `c238d19c2116af4210508746fda88b9ca155bc50f370ae4c6e7ffd534998d824` |
| `scripts/extract_pwc_bundle.py` | `8eaede1f72f60ef12644729e94e958181f833ddc789e13721f53acb718e7e1f9` |
| `tests/test_extract_pwc_bundle.py` | `3e0faae171a310c44f75c41e550235d54770ba0b36754a80b72b09504d466224` |
| `tests/test_stage4_miner_launch.py` (new) | `31a172fed5a3e86a1a4c4c5cd9142978401c3368ce22621d78b42d5794b3cbd5` |
| `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff` | `4c2d41c85f88b00f034167a9085b166c1dc1e17bf7e9d899c642cc12daa8bc50` (applied: launcher `fe964b04fe3ea431986835ff2fff934349864751f8974da5ccebb6b1a62bf5d3`) |
| `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | `c19b58640d00d4693523f458d3efb3a9496c0a1e43188895f26496baeabaaad1` |

**`/home/michael/tfm_worktrees/pwc-offset` (agent 2):**

| File | sha256 |
|---|---|
| `scripts/extract_miner_bundle.py` | `2f1d35d450150f75ae78dc70bc5b34a452045b320d0552f7f7ff9f9e254115dd` |
| `tests/test_extract_miner_bundle.py` | `4e4d390f1caab48e6112c96c5ac520dd849786d3177ddfd5a8b37a5935b7b312` |
| `docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md` | `9e7cf7b2916f6471a1092dd1d58dd5d0e39f8cb59783987812cb6a7c8c9440c8` |

**Agent 2's files in `pwc-backup` are current.** The Alpha lead refreshed all three from `pwc-offset` before the
landing, byte-identical, at the hashes in the table above. The R1 attachments `4c27042f…ee611` and `ed705ff8…61347`
are gone, and the landed `6eafad9` carries the current versions.

## Test status (CPU, no run)

- **Agent 1's suites in `pwc-backup`,** run one after another:
  - `test_compare_engine_outputs.py`: 36/36;
  - `test_extract_pwc_bundle.py`: 32/32;
  - `test_stage4_miner_launch.py`: 14/14.
- **The Alpha lead's run:** all four test files exit 0 in a fresh scratch copy with agent 2's `2f1d35d4…` against
  comparator `040f77b6…`. That is the lead's report; agent 1 did not repeat it.
- **Since the draft:** no code changed. The runbook hash above is the landed one.
- **No real Stage 4 run has happened.** The B6 publication-lag bound, real WAL states and real engine output are
  unverified. Equality of engine outputs is an engineering check; it says nothing about pool hit rate against `k/1000`.

## Positions table (for the record; allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

| Question | Position | Severity |
|---|---|---|
| R2-Q2 Job identity and credit requirement (your Q2) | | |
| R2-Q3 Coverage and link cross-identifiers, both extractors (your Q3) | | |
| R2-Q5 PWC extractor: WAL, accounting, threshold, link, membership, out-dir (your Q5) | | |
| R2-Q6 Miner receipt, B1-B8, runbook step B4a (your Q6) | | |
| R2-Q7 `optuna_studies/` and the halt flag (your Q7) | | |
| R2-Q8 Derived launcher: params gate, no `--clear-halt`, sampler not run (your Q8) | | |
| R2-Q9 Regressions added (your Q9) | | |
| R2-F1 Prior-record references (your Finding 1) | | |
