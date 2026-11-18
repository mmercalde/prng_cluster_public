# CLAUDE CODE REPORT — DOC FIX LIST, AGENT A (framing documents) — 2026-11-17

**Brief:** `/home/michael/doc_fix_brief_20261117.md`, Agent A: items A3, A4, A5, B1, B2, B3, B4, D4
of `docs/DOC_FIX_LIST_20261113.md`. **Worktree:** `/home/michael/tfm_worktrees/docs-a`, detached at
`03f3712`. Nothing committed, pushed or run on a GPU or rig. `~/miner_staging` not opened.

## Prior record

*Placed first under `CLAUDE.md` "Prior record first" (owner rule, 2026-11-13). The brief asks for
the plain summary first; it follows directly below. See Question 6.*

**Searches run** (2026-11-17, in this worktree at `03f3712`; `/bin/grep` because the shell `grep`
skips ignored files):

```
cd docs; S="PROJECT_FILE_CATALOG.md TB_RULING_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange"
for pat in "Seed Reconstruction" "Reverse-engineer" "MATHEMATICAL_WHITEPAPER" "F-7" \
           "KNOWN_ANSWER_VALIDATION_INVENTORY" "Paradigm_Correction" "SESSION_CHANGELOG_20261113"; do
  /bin/grep -rl -i -- "$pat" $S; done
/bin/grep -n -i "reconstruct|reverse-engineer|true seed|recover" README.md docs/README.md docs/proposals/README.md
/bin/grep -n "D16" docs/PIPELINE_BEHAVIOUR_MODEL.md
/bin/grep -rn -i "whitepaper[^|]{0,40}`:[0-9]|WHITEPAPER.md:[0-9]" --include=*.md .      # who cites whitepaper lines
/bin/grep -rnoh "[a-z/]*README\.md:[0-9]+" --include=*.md .                             # who cites README lines
/bin/grep -n -i "pa_sieve|planted|forward|reverse|recover|bidirectional" docs/SESSION_CHANGELOG_20260314_S143.md
/bin/grep -n "\[::-1\]" miner/range_miner_worker.py        # and the same at 048fd47~1 and b10e1ef
git log -- docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md; git show --stat 689f475
git log --since=2026-11-13T00:00 --until=2026-11-14T00:00; ls docs | /bin/grep 20261113
```

Also read in full: the four read-first documents of the brief, `CLAUDE.md`, `AGENTS.md` §1,
`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md`,
`docs/TFM_SESSION_HANDOFF_20261113.md`, the whitepaper, Chapter 2 §3 and §12-§14.3 and its
Version History, `tests/test_chapter2_content_gate.py` (the anchor and closure gates).

**Documents found that already decide, authorise or record the subjects:**

| Subject | Document and section |
|---|---|
| All eight items | `docs/DOC_FIX_LIST_20261113.md:16-18, 25-28, 48` |
| README framing (A3, A4) | `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1 `:1105-1108` and `:1192` (D16, "recorded not fixed"); `docs/PROJECT_FILE_CATALOG.md` §6.4 `:1004-1007` ("Noted for Alpha — not fixed here") |
| Whitepaper vs implementation (B1, B2) | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §3.5 and F-7 (`:1175`, `:1387`); `docs/TODO_CONSOLIDATED.md:146` (F-7, REVIEW); `docs/CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md:508` (D10) |
| 2026-11-13 reverse-sieve status (B2) | `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md` §7 `:89-101`; realignment note §3 `:38-50`; Beta R2 §1 `:24-50` and §5 `:138-150` |
| PA harness claim (B3) | `docs/SESSION_CHANGELOG_20260314_S143.md:61-72`; `docs/CHAPTER_4_AUDIT.md:139` (cites the inventory's §5, not `:97`) |
| Paradigm proposal status (B4) | commit `689f475` (2026-01-19); `docs/PROJECT_FILE_CATALOG.md:610`; Beta R2 `:27-28` |
| 2026-11-13 changelog (D4) | `docs/TFM_SESSION_HANDOFF_20261113.md:180` (listed as owed) |

**Labels:** every item worked here is `ALREADY RECORDED` in the fix list; the work is the recorded
fix. No item is `CHANGES A RULING`. One sub-item is `NEW`: the S143-changelog vs harness-code
contradiction on how the PA harness's reverse works (B3 row). The F-7 anchor drift
(`range_miner_worker.py:888` → `:1330`) is `NEW` only as a line number.

**Not searched:** the host, ser8 archives, the private repository, claude.ai project files and
the current claude.ai Project description/instructions text (A5), the four project PDFs.

---

## Summary

1. All eight Agent A items were open at `03f3712`; seven are now done in place, and B4 has a note only, because its adoption date has to come from Michael.
2. No line number moved in any edited file, except Chapter 2 after `:1427` (its Version History). The whitepaper, the three READMEs, the inventory and the proposal are the same length as before, because the line anchors other documents cite stay valid.
3. B2 records the 2026-11-13 reverse-sieve fact as status with Beta's limits, and leaves out the pitfall doc's `:97-98` bound, which Beta R2 §5 says is false.
4. Gate: the script form passes 12/12 (`SENTINEL : PASS`) before and after. The brief's `pytest` command collects no tests and exits 5 both times.
5. Seven questions for Michael. The out-of-scope items include one CODE framing hit (`agents/prompts/global_mission.py:6`).

## Items

| Item | State found at `03f3712` | What changed (file:line) | Source |
|---|---|---|---|
| **A3** | Open: `README.md:4` "reverse-engineering PRNG behavioral patterns"; `docs/README.md:11` "Reverse-engineer PRNG behavior" | `README.md:4-5`: functional mimicry of PRNG surface output, black box, not seed recovery or state reconstruction, judged by pool hit rate vs `k/1000` on unseen draws. `docs/README.md:11`: same statement. Line counts unchanged (82, 339) | fix list A3 `:16`; `PIPELINE_BEHAVIOUR_MODEL.md:1105-1108` (D16); `PITFALL_STATE_RECOVERY_YARDSTICK.md:11-16` |
| **A4** | Open: `docs/proposals/README.md:1` "…& Seed Reconstruction System" | `:1` retitled "…& Functional Mimicry System" (the title `docs/README.md:1` already uses). Also `:11`, which carried the identical A3 sentence, now has the A3 wording. Line count unchanged (299) | fix list A4 `:17`; D16 `:1105`; catalog `:1004-1007` |
| **A5** | Open; the current Project text is not in the repo | New `docs/DRAFT_PROJECT_DESCRIPTION_20261117.md`: prior record; a short description; full instructions text; notes. Michael pastes it | fix list A5 `:18`; `CLAUDE.md` header, "The yardstick", "Prior record first" |
| **B1** | Open: no header note; §4 `:57-59` backward index; `:112` "only the true seed survives"; `:140` `P(s = s*)` | `:2` header note (AI formalisation; Michael's intent governs; yardstick is hit rate; §4/§6/§8 do not describe the implemented method; §7/§10 `s*` and "That is why this system works" flagged as unmeasured). Marks on `:53` (§4 heading), `:112`, `:143`. **Maths not rewritten.** Each note is on an existing line, so the 167 lines and every cited anchor (`:31`, `:57-62`, `:79`, `:116-131`, `:158-167`, …) are unchanged | fix list B1 `:25`; `PITFALL…:27-29`, §1, §3, `:74-77`; `miner/range_miner_worker.py:1330` |
| **B2** | Open: §3.3 table unqualified; §3.5/F-7 hold no 2026-11-13 measurement; F-7 says "named, not resolved" | `:296` note after the §3.3 table (same stream, newest-first target, not a backward walk). `:329` §3.5: the 2026-11-13 status as fact, with "CPU only, not GPU or VM101", Beta's limits quoted, labelled THEORETICAL, the hit-rate prediction and the measurement that would settle it, and a cross-reference to the pitfall doc. `:1175`, `:1387`: "confirmed, unresolved" appended, old wording kept, reverse slice now at `:1330`. Version History 4.2.1 at `:1428-1440` | fix list B2 `:26`; `PITFALL…:54-56, 91-101`; realignment note `:40-50`; Beta R2 `:34, 38-46`, §5 |
| **B3** | Open: `:97` "proves the bidirectional sieve method recovers a planted seed" | `:97-98` corrected: S143 records a forward-sieve known-answer check only (`SESSION_CHANGELOG_20260314_S143.md:69-72`); the bidirectional claim is marked **UNVERIFIED**. Also recorded as fact: S143 `:64-66` says the harness reverse is the forward kernel on `draws[::-1]`, but `pa_sieve_validation_harness.py:64-79` steps backward by modular inverse. Line count unchanged | fix list B3 `:27`; S143 changelog; `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md:37`; Beta R2 `:48-50` |
| **B4** | Open: `:5` "PENDING APPROVAL" | **Status line not changed** (needs Michael's date). `:6` status note: the fix-list instruction; commit `689f475` (2026-01-19, Michael) edited the Chapter 13 documents named in §4 (whether every §4 item was applied was not checked); the documents that cite this one as the origin of the doctrine | fix list B4 `:28` ("with the date Michael gives") |
| **D4** | Open: no `SESSION_CHANGELOG_20261113_YARDSTICK.md` | New `docs/SESSION_CHANGELOG_20261113_YARDSTICK.md`. It is reconstructed from commits `6267c7b`, `4b1e81e`, `d60a714` and the committed documents, and says plainly that it is not a first-hand log | fix list D4 `:48`; handoff `:180` |

Also written: `docs/SESSION_CHANGELOG_20261117_DOC_FIX_A.md` (`CLAUDE.md` §1.8).

**Verification:**
- Before and after, `python3 -m pytest tests/test_chapter2_content_gate.py -q` printed "no tests ran" (exit 5).
- Before and after, `python3 -u tests/test_chapter2_content_gate.py` gave 12/12 green and `SENTINEL : PASS` (exit 0).
- `git diff -U0` hunk headers confirm the line-count claims above.

## Questions for the owner

1. **B4:** on what date was `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md` adopted? Is it 2026-01-19, the date of `689f475`, or another date? When you give it, `:5` changes to "ADOPTED <date>".
2. **The gate command:** `pytest` collects nothing from `tests/test_chapter2_content_gate.py` and exits 5. The script form is the real gate. Should briefs name the script form, or should the file get a pytest wrapper? A wrapper is a test-code change and is out of scope here.
3. **B1:** do you want whitepaper §4, §6 and §8 rewritten to match your method? The brief and the fix list both say not without you. The notes only mark them.
4. **B1/B2 format:** I put the notes on existing lines (one long header line in the whitepaper, a blockquote in place of a blank line in Chapter 2) so the line anchors cited by Chapter 2, the fact file and about eight reports still resolve. Is that acceptable? The alternative is normal multi-line notes plus re-anchoring the citing documents.
5. **F-7 wording:** Chapter 2 now says both "named, not resolved" (as before) and "confirmed, unresolved" (the fix list, the pitfall doc `:99` and Beta R2 `:44`). Should the older wording be retired?
6. **Report order:** the brief puts the plain summary first. `CLAUDE.md` requires `## Prior record` before anything else. I followed `CLAUDE.md` and put the summary directly after it. Which order do you want?
7. **A5:** please compare the draft with the current claude.ai Project text, which I cannot read, so that nothing correct is lost.

## Out of scope / not done

**CODE or JSON items on the fix list (not touched):**
- **C1:** `prng_registry.py:3911`, `:3917`, "backward validation" strings.
- **C2–C4:** registry and manifest JSON. Agent B writes the proposed wording.
- **E1–E3.**

**A related CODE hit (not on the fix list):** `agents/prompts/global_mission.py:6`. The agents' mission text says the system "reverse-engineers" PRNGs (`docs/CHAPTER_10_AUDIT.md:208`). This is the same framing as A3, but in a prompt the agents read.

**Agent B's items (not touched here):** A6, C5, C6, D2, D3. Two notes for B:
- `docs/PROJECT_FILE_CATALOG.md` §6.4 `:1004-1007` and `docs/PIPELINE_BEHAVIOUR_MODEL.md` D16 (`:1105-1108`, `:1192`) still say the README framing is "not fixed". They now need a status update.
- D3: the catalog needs the new files from this pass.

**Other items on the list:**
- A1 and A2 were already done at `6267c7b`: `CLAUDE.md:17` and `AGENTS.md:19-36` carry the yardstick blocks, and the pitfall doc exists.
- D1, D5 and D6 were not assigned to Agent A.

**Found but not on Agent A's list (not changed):**
- `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:91` cites `miner/range_miner_worker.py:1201`. That was correct at `b10e1ef`; the line has been at `:1330` since `048fd47`.
- The same pitfall doc's `:97-98` bound is the one Beta R2 §5 calls false. The handoff `:181` lists it as owed.
- Chapter 2 cites older `range_miner_worker.py` lines (`:888` in §3.2 and F-7, `:116` for `is_reverse_family`, now `:215`). The content gate checks only that anchors are in range, not that they point at the right line.
- "Seed reconstruction" and "Reverse-engineer" framing also appears in:
  - `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md:17`, `:19`
  - `docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md` and `_v3.md`
  - `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2 .md` and `_v3_9.md`

**Process notes:**
- Early in the session I piped two short read-only commands to `tail`, which breaks `CLAUDE.md` hard rule 6. Neither was a run. After that I stopped using pipes on test runs.

---

## Addendum — owner answers and lead decisions (2026-11-17 17:17 PST)

**Answers received:**
- Q1/B4 (owner): mark the proposal ADOPTED 2026-01-19, the date of `689f475`.
- Q3/B1 (owner): notes only; no whitepaper section is rewritten.
- Q2 (lead): use the script form of the gate in future.
- Q4 (lead): the note format is accepted.
- Q5 (lead): use "confirmed, unresolved" for F-7 everywhere, keeping one line that records the change.
- Q6 (lead): follow the `CLAUDE.md` order.
- Q7 (lead): the owner compares the A5 draft himself.

**Applied with the Edit tool:**
- **Q1:**
  - `docs/PROPOSAL_Documentation_Paradigm_Correction_v1_2.md:5` now reads "ADOPTED 2026-01-19". It cites `689f475` and this ruling, and records that it was "PENDING APPROVAL".
  - The `:6` status note no longer says the line is waiting for a date.
  - File length unchanged (182).
- **Q5:**
  - `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1175` (§12) and `:1387` (§14.3) now say "confirmed, unresolved". `:329` (§3.5) already did.
  - `:1175` keeps one line: "This wording replaced 'named, not resolved' on 2026-11-13".
  - Version History 4.2.1 is updated at `:1436-1437`.
  - No line before `:1428` moved.
  - "named, not resolved" now appears only in that one line and in the Version History entry.

**Gate:** `python3 -u tests/test_chapter2_content_gate.py` gives 12/12 green, `SENTINEL : PASS`, exit 0.

**Not changed (outside these two files):** `docs/TODO_CONSOLIDATED.md:146` still says F-7 is "named not resolved". It is not an Agent A file. To make the wording consistent everywhere, that row needs the same change.
