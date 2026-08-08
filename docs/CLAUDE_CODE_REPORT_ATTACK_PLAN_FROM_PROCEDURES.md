# CLAUDE CODE REPORT — ATTACK PLAN FROM THE OFFICIAL DRAW PROCEDURES

**Date:** 2026-08-08 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:**
`/home/michael/distributed_prng_analysis` · **HEAD:** `8bbe79e`
**Type:** read-only analysis producing a **proposal**. Nothing was launched, no production code was
written, no configuration was changed, nothing was committed.
**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_ATTACK_PLAN_FROM_PROCEDURES.md`

## Source document — identity and readability

```
path    docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf
size    666,629 bytes          sha256  7048b2552ee22ba09d4c0e3d5481af9d280992331de755f2273e7e1f2a792f74
pages   23 (pdfinfo)           PDF 1.7, unencrypted, text-extractable
title   "Daily and Superlotto Plus Draw Procedures"   author "Ca Lottery"
effective  June 9, 2021        header "MODIFIED for Release for Solicitation"
```

Read in full this session via `pdftotext -layout`. `file(1)` reports "2 pages"; `pdfinfo` reports
23 and the extraction yields 23 numbered pages ("Page 1 of 22" … "Page 22 of 22" plus an unnumbered
cover). **`pdfinfo` is authoritative; `file`'s page count is wrong.**

> **Terminology.** Per `CLAUDE.md`, this project is TFM and does not use the operator's category
> noun. The source document's own title and quoted text are reproduced verbatim where citation
> requires it; everywhere else this report says "the procedures document", "the operator", "the
> draw".

## ⚠ Citation hazard — the document's own section numbering is internally inconsistent

Established by reading, and it affects every citation in this report:

- **`XIV` is used twice** — "VERIFICATION OF SLP DRAW CLOSED STATUS AND JACKPOT AMOUNT" (p.16) and
  "DRAW EXCEPTIONS AND ANOMALIES" (p.22, where sequence requires `XXIV`).
- **Cross-references disagree with the body.** The text says both *"See XXIII. DRAW EXCEPTIONS AND
  ANOMALIES"* (p.4, p.15) and *"See XXIV. DRAW EXCEPTIONS AND ANOMALIES"* (p.5); `XXIII` is
  actually "DRAW ROOM EXIT" (p.21).
- **The SLP block's internal references are uniformly +1 off** — `XIV`→"See SECTION VII" (closed
  status is `VI`), `XVIII`→"See SECTION IX" (results entry is `VIII`), `XIX`→"See SECTION X"
  (PWS is `IX`), `XX`→"See SECTION XI" (e-mailing is `X`).
- `XVI` contains **two steps numbered 4** and skips no others.

**Therefore every citation below carries section *and* page**, e.g. `§V.14 (p.6)`. A section number
alone is not a reliable locator in this document.

## Marking convention

The brief requires document-derived, inference and speculation not to blur. Implementation claims
are a fourth, separately-evidenced class:

| mark | meaning |
|---|---|
| **[DOC]** | **derived from the procedures document** — quoted or directly restated, with §/page |
| **[INF]** | **inference** — a conclusion drawn from [DOC] plus code/data facts. Premises always stated |
| **[SPEC]** | **speculation** — plausible, not supported by the document or by evidence gathered |
| **[CODE]** | verified live in this tree at `8bbe79e` this session, `file:line` |
| **[DATA]** | measured live from `daily3.json` this session (gitignored; invisible to clone-scoped audit) |

"The document does not specify" is used literally and is itself a [DOC] statement.

---

# PART A — WHAT THE DOCUMENT ACTUALLY SPECIFIES

## A.1 Which games are drawn in which session

**[DOC]** DRAW TIMING table (p.2), rows verbatim:

| Draw | Room Entry | Verify Open | Pre-Test | Verify Closed | Conduct Live | Room Exit |
|---|---|---|---|---|---|---|
| **Mid-day D3** | 11:45am | 12:00pm | **12:30pm** | 1:00:10pm | By 1:10pm | By 1:45pm |
| **Evening D3, D4, F5, DD** | 4:45pm | 5:00pm | **6:00pm** | 6:30:05pm | By 6:40pm | By 8:00pm |
| **SLP** | 4:45pm | 5:00pm | SLP by 6:00pm; Eve draws @ 6:00pm | 7:45:00pm | By 8:15pm | By 11:00pm |

**[DOC]** Midday draws **Daily 3 alone**. Evening draws **D3, D4, F5 and DD** in one session. The
midday and evening sessions have **separate draw-room entries and a sealed exit between them**
(§XXIII.5, p.22: *"Seal the door with the numbered seal indicated in the Log Book"*).

**[DOC] The document does not state the order in which the four evening games are drawn.** §V.12
(p.6) lists the specs in the order *Fantasy 5, Daily 4, Daily 3, Daily Derby*, but that is a
verification list, not a stated execution order.

**[DOC]** SuperLotto Plus is normally drawn on **manual ball machines** (§XI–XIII, XV, pp.12–17):
serpentine ball loading, mixing chambers, an Independent Designated Drawer who *"presses the button
on the manual drawing machine"* (DRAW TEAM MEMBERS, p.1). It reaches the automatic machine **only**
as a fallback — §XI.16 Note (p.13): *"If the manual draw machines are inoperable or unavailable,
proceed with using the automatic draw machine for the SLP draw."*

**[INF]** Therefore **SuperLotto Plus does not normally consume the automatic machine's generator
at all**, and does not enter the between-Daily-3 accounting on Wed/Sat. Premise: the manual machines
are physical ball machines with no RNG, and the document routes SLP to them by default.

## A.2 The draw specs — digits, ranges, replacement, and output count

**[DOC]** §V.12 (p.6), verbatim:

```
CA Fantasy 5            05:01-39u
CA Daily 4              04:00-09r
CA Daily 3              03:00-09r
CA Daily Derby          03:01-12u 03:00-09r
```

**[DOC]** §XVI.10 (p.18), for SLP on the automatic machine: `SuperLotto Plus  05:01-47u 01:01-27u`.

**[DOC] The document never defines the notation, and never defines `u` or `r`.**

**[INF]** The format is `CC:LO-HI<mode>` — count, inclusive range, mode — with **`u` = unique (no
repeats within the selection)** and **`r` = repeats permitted**. Premises, all internal to the
document: (i) every count/range pair is consistent with the count fitting in the range only when
`u` is used, and `r` appears exactly where the range is `00-09` and the count exceeds 10 only under
repetition; (ii) `05:01-39u` and `05:01-47u 01:01-27u` are two-part specs whose second part has
count 1, where uniqueness is vacuous — consistent with a mode flag applied uniformly rather than a
meaningful choice; (iii) Daily Derby's two-part spec `03:01-12u 03:00-09r` pairs a
distinct-selection component with a digit component. No external source was used.

**[INF] Selections per draw of each game**, from the specs:

| game | spec | selections |
|---|---|---|
| Daily 3 | `03:00-09r` | **3** |
| Daily 4 | `04:00-09r` | **4** |
| Fantasy 5 | `05:01-39u` | **5 minimum** |
| Daily Derby | `03:01-12u 03:00-09r` | **6 minimum** (3 + 3) |
| SLP (automatic path only) | `05:01-47u 01:01-27u` | **6 minimum** (5 + 1) |
| **evening set total** | | **18 minimum** |
| **midday set total** | | **3** |

**[DOC] How many generator outputs a selection consumes is not specified anywhere in the document.**
The document describes the operator's procedure, not the machine's sampling algorithm.

**[INF]** Two consequences follow, and they are the crux of Part C:

1. For **`r` games** (D3, D4, and DD's second component), selections are independent and the
   selection count is **fixed**.
2. For **`u` games** (F5, DD's first component, SLP), the count is **fixed at the minimum if the
   machine samples without replacement directly** (partial shuffle, or index-into-remaining), and
   **variable and unbounded above if it samples with rejection**. The document does not say which.
   Under rejection sampling the expected selection counts are
   `39/39 + 39/38 + … + 39/35 ≈ 5.28` for F5 and `12/12 + 12/11 + 12/10 ≈ 3.29` for DD's horse
   component. **The published values cannot distinguish the two mechanisms** — a rejected duplicate
   leaves no trace in the final set.

## A.3 Equipment selection (§II) — and what is consequently unrecorded in public results

**[DOC]** §II (p.4), verbatim: *"A random number generation (RNG) program is used to select the
primary and alternate draw equipment which will be used for the draw."*
§II.1: *"Access the appropriate tab in the RNG program on the PC desktop to select the draw
equipment to be used for the draw(s). Auditor Verifies"*
§II.5: *"Record the draw equipment selection(s) on the Draw Games Certification."*

**[DOC] There are exactly two automatic draw machines.** §VII.5 NOTE (3) (p.9) and §XVII.6 NOTE (3)
(p.20): *"If **both** of the automatic draw machines malfunction, see XXVII. EMERGENCY OFF-SITE
DRAWS."* §VII.5 NOTE (1) and (2) both direct the DC to *"use the alternate automatic draw machine"* —
the singular alternate.

**[DOC]** The selected machine's identity **is recorded**, on retained paperwork only:
§VII.7 Note (p.9) — *"Automatic draw machine number is the last digit on the automatic draw machine
serial number located at the top left of Official Drawing Record form"* — and §II.5's Draw Games
Certification. §VII.11 (p.9) distributes copies to the auditor and Internal Audits; **none of the
publication steps carries it.** §VIII (p.9-10) transmits only *"the official draw results"* to IGT;
§IX (p.11) verifies the public website shows *"draw results, prize amounts, and winning city(ies)"*.

### A.3.1 Per draw, or per draw session? — adjudicating the open item

`docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md:620-643` records this as *"the single
highest-leverage open item the search surfaced"* and could not settle it, because the PDF was not
in the repository: §II says *"the draw"*, Alpha's gloss in
`docs/TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md:24-27` says *"per draw session"*, and
S112 (`:139`, `:181`) recorded the **per-draw** reading. The whole per-session governance line
descends from the gloss.

**With the document in hand: the gloss is correct. Equipment selection is PER DRAW SESSION.**
**[INF]**, on four [DOC] premises:

1. **§II is positioned once per draw-room session.** The procedure runs DRAW ROOM ENTRY → §I
   recording → §II selection → §III open status → §IV system status → §V setup+pre-test →
   §VI closed status → §VII live draws → … → §XXIII exit and re-seal. The DRAW TIMING table gives
   **one entry and one exit per session**, and the room is sealed between them (§XXIII.5, p.22).
   §II cannot execute per game without re-entering a sealed room.
2. **§II.1's own object is plural** — *"the draw equipment to be used for the draw(**s**)"*.
3. **§V.8 (p.6) selects one "game set"** — *"Select a game set from the drop-down list"* — on the
   one machine powered on at §V.2, under one Operator/Auditor login (§V.9-10).
4. **§VII.2 (p.8) conducts the whole set with one action** — *"Conduct the live draw(**s**) by
   clicking the green [Run LIVE (#) Draw] button"*, where `(#)` is a count placeholder.

**[INF]** §II.5's *"selection(s)"* plural is explained without a per-draw reading: on Wed/Sat the
same §II run also selects the SuperLotto and Mega **machines and ball sets** (§XI.1, §XI.11, pp.12-13)
— several equipment *items*, one draw session.

**Consequence for governance:** the *"combined-session sieve non-certifying"* / *"production
re-optimization is per-session"* ruling chain (skill §2.10b) rests on a premise this document
**confirms**. **[INF]** It does **not** follow that per-session scoping is sufficient — see B.5 and
Part C.4, because the machine is re-selected at *every* session, including consecutive sessions of
the same series.

## A.4 The two RNGs per machine (§V.4)

**[DOC]** §V.4 (p.6) and §XVI.3 (p.17), verbatim: *"Verify both A and B RNG icons displayed are
green."* That is the **entire** treatment of A and B in the document.

**[DOC]** The document does **not** say what A and B are, whether one is primary and the other a
standby, whether the machine alternates between them, whether they are seeded independently, or
which produced any given value. It states only that both must be green before the session proceeds.

**[INF]** A published value therefore cannot be attributed to A or B from any public artifact — the
distinction is not recorded even on the retained Official Drawing Record, which §VII.7 describes as
carrying date, results, machine number and draw numbers.

**[SPEC]** A green/green requirement is the shape of a redundant pair with a health check, which
would be consistent with either hot-standby or alternation. Nothing in the document distinguishes
them and no weight should be placed on either.

## A.5 The pre-test (§V.14-16) and the `Run Draw as Test` flag — THE COUNT, SETTLED

**[DOC]** §V, verbatim, in order (pp.6-7):

- §V.7 — *"click the [Start New Draw Session] button on the Draws Controls section"*
- §V.8 — *"Select a game set from the drop-down list. Auditor Verifies"*
- §V.12 — *"Before starting the draw session and the Pre-Test, enter draw number(s) in the draw
  number line of the draw info box and verify the draw specs for each game are correct."*
- §V.14 — *"After entering draw numbers, click the green [Start Draw Session] button to **run
  Pre-Test**."*
- §V.15 — *"Click [Display Results] to see and hear the video. Verify the spoken results agree with
  the selected numbers."*
- §V.16 — *"After the pre-test draw(s) is complete, **remove the check mark (✓) next to "Run Draw as
  Test"** to run an official draw. Auditor Verifies"*
- §V.16 Note — *"**Document pre-test details if an additional pre-test(s) are conducted for any
  reason; a Draw Anomaly report is required.**"*
- §VII.1 (p.8) — *"Prior to starting the official draw(s) verify the check mark (✓) next to 'Run
  Draw as Test' is un-checked."*

### The citations skill §0.4 has been carrying as `UNAVAILABLE`

Skill §0.4 states: *"One automatic pre-test session runs before an automatic Daily draw on the
selected equipment (§V: Pre-Test via `[Start Draw Session]`). Additional pre-test draws run only
when an anomaly requires them,"* with *"Citation `UNAVAILABLE` — the PDF is not in the repo,"* and
records the 2026-08-01 correction of an earlier *"two pre-test draws before every live draw"*
misreading. **Both halves are confirmed, and the real anchors are:**

| skill §0.4 claim | anchor now available | verdict |
|---|---|---|
| **one** automatic pre-test session before an automatic Daily draw | **§V.14 (p.6)** — one `[Start Draw Session]` click, *"to run Pre-Test"*, singular | **CONFIRMED [DOC]** |
| additional pre-tests only when an anomaly requires them | **§V.16 Note (p.7)** — *"Document pre-test details if an additional pre-test(s) are conducted for any reason; a Draw Anomaly report is required"* | **CONFIRMED [DOC]** |
| the "two test draws" language belongs to **manual SuperLotto Plus equipment** | **§XII.1 (p.13)** — *"Conduct **two test draws on each machine**"*, under the heading **"SLP TEST DRAWS AND ANALYSIS OF RESULTS"**, following §XI's ball-loading setup | **CONFIRMED [DOC]** |
| the automatic path takes one pre-test | **§XI.16 Note (p.13)** — *"**Only one pre-test is necessary, as the draw is automated.**"* | **CONFIRMED [DOC], explicit and decisive** |
| pre-test outputs generated, verified, never published | §V.15 verify; §VII.5/§VIII publish only official results | **CONFIRMED [DOC]** |

**§XI.16's sentence is the single strongest line in the document for TFM's purposes** — it states
the automatic pre-test count explicitly and contrasts it with the manual two-test-draw rule in the
same breath. **Skill §0.4's `Citation UNAVAILABLE` can be replaced with `§V.14, §V.16 Note, §XI.16
Note, §XII.1 (pp.6-7, 13)`.** That is a documentation update for the owner, not a change this
report makes.

### How many outputs a pre-test consumes

**[INF]** The pre-test runs **the same game set** as the live draw: §V.8 selects the set once,
§V.12 verifies the specs of *"each game"* once, §V.14 runs the pre-test, and §V.16 unchecks a
**flag** — no re-selection of game set, no re-login, no new session. **The pre-test and the live
draw are the same draw session on the same machine, separated only by a checkbox.**

**[INF]** Therefore a session consumes **two full game sets** — pre-test then live — with the
pre-test's outputs unpublished. This is the physical basis of TFM's `skip`, now anchored.

**[DOC] The document does not specify** whether the `Run Draw as Test` flag has any effect on the
generator, nor whether `[Start New Draw Session]` (§V.7) versus `[Start Draw Session]` (§V.14)
implies any generator reinitialisation. Both button names appear without definition.

## A.6 Power-on and `[Shut Down]` — what the machine does at start-up

**[DOC]** §V.2 (p.6) — *"Retrieve the keys from the secure location and **push left power button**
on the selected automatic draw machine."*
§V.3 — *"After powering on the selected machine, ensure it audibly states, '**Welcome to [machine
name], RNS on line.**'"*
§VII.8 (p.9) — *"**Power off** the automatic draw machine by clicking the **[Shut Down]** button
then click 'Yes' when the option appears."*
§V.6 — *"Verify the date and time are correct on the lower screen."*

**[DOC] The document specifies a full power cycle per draw session and says nothing whatsoever
about generator state across it.** There is no statement about seeding, reseeding, entropy,
persistence, or continuity. The only start-up assertion is the spoken banner and the A/B green
check (§V.4).

**[INF]** The observable consequence is a **hard, twice-daily boundary** at which the model must
either assume state persists across a power cycle or admit a discontinuity. Nothing in the
document supports the assumption; nothing contradicts it.

**[SPEC]** *"RNS on line"* at power-on and a verified system date/time (§V.6) are consistent with a
time-seeded initialisation, which would make every session an independent stream. This is
speculation and is offered only because the alternative — silent persistence across power-off — is
equally unsupported and should not be adopted by default either.

## A.7 The vendor stack — what identifies the generator family, and what does not

**[DOC]** The only name the document gives the generator subsystem is **"RNS"**, in the power-on
banner *"Welcome to [machine name], RNS on line"* (§V.3, p.6; §XVI.2, p.17). **The document never
expands the abbreviation, never names an algorithm, and never names the vendor of the automatic
draw machine.** The machine name itself is redacted to a placeholder.

**Three names in the document that must NOT be read as the generator's vendor:**

| name in document | what the document actually says it is |
|---|---|
| **IGT** (International Game Technology) | the **online/gaming system** — draw summaries and open/closed status (§III), primary and secondary *gaming* systems (§IV.2, p.5), winning-number entry and verification (§VIII, pp.9-10), share values. §IV is titled "VERIFICATION OF GAMING SYSTEM STATUS". **[DOC] It is never described as supplying, hosting or operating the draw machine's RNG.** |
| **"Criterion drawing machines"** (§XXI.5, p.21) | the **manual SLP ball machines** left in the secured draw room — §XXI is "SECURING THE SUPERLOTTO PLUS DRAW EQUIPMENT". **[DOC] Not the automatic draw machine.** |
| **the "RNG program on the PC desktop"** (§II.1, p.4) | the **equipment-selection** tool on a desktop PC. **[DOC] A different program from the draw machine's own RNS**, used before the machine is even powered on (§V.2). |

**[DOC] Per the brief's instruction, no vendor or algorithm is guessed from outside the document.
The document identifies no generator family.**

**[INF] This is a direct hit on TFM's PRNG-family choice.** The sieve targets `java_lcg` — a 48-bit
LCG, `m = 0xFFFFFFFFFFFF` **[CODE]** `prng_registry.py:969`. **The procedures document provides no
support for that choice, and no support for any of the other 43 registry families either.** It is
not evidence against; it is an absence of evidence. Anyone reading the document expecting it to
justify the family will not find it.

## A.8 Anomaly paths that inject extra, unpublished draws

**[DOC]** Every path below injects generator consumption that never appears in public results.

| # | path | § / page | what is injected |
|---|---|---|---|
| **N1** | additional pre-test(s) *"conducted for any reason"* | §V.16 Note, p.7 | **one or more extra full pre-test sets** |
| **N2** | DC clicks `[Run LIVE (#) Draw]` before pools close: *"let the invalid live draw proceed … Reset the system so the DC and Auditor can log back on, **run a test draw**, then wait for pools to close before conducting the live draw"* | §V.16 Note, p.7 (repeated §XVI.15 Note, p.19) | **an entire extra LIVE set + an extra test set**, printed and retained marked *"Not a valid draw – pools not yet closed"*, **never published** |
| **N3** | draw PC malfunctions with all numbers selected, reports unprintable: *"Re-boot … and conduct **another pre-test and another official draw**"* | §VII.5 NOTE (1), pp.8-9 | **an extra pre-test set + an extra live set**, plus a reboot |
| **N4** | draw PC malfunctions before all numbers selected: *"The numbers that have been selected are not valid"* | §VII.5 NOTE (2), p.9 | **a partial set**, of unknown length |
| **N5** | alternate machine substitution — *"use the alternate automatic draw machine to conduct the aborted draw"* | §VII.5 NOTE (1) and (2) | **a machine change mid-session** — a stream change, not a skip |
| **N6** | both machines malfunction → §XXVII EMERGENCY OFF-SITE DRAWS, at the Operational Recovery site under a **separate procedures document not included here** | §VII.5 NOTE (3), p.9; §XXVII, p.22 | **entirely unspecified** |
| **N7** | DVM failure, draw-room seal mismatch, IGT access failure | §I Note p.3; DRAW ROOM ENTRY 3b p.3; §XXVI p.22 | no generator effect stated |

**[DOC] How often the document expects the stream to be perturbed: it does not say.** §XXIV
[misnumbered `XIV`] (p.22) states the opposite of a rate — *"**No specific course of action can be
prescribed since every incident is potentially unique.**"* Every path requires a Draw
Anomaly/Exception Report distributed by e-mail; **none of those reports is a public artifact named
anywhere in the publication steps (§VIII, §IX, §X).**

**[INF]** The perturbation rate is therefore **not merely unknown — it is unobservable from
published results**, and no amount of modelling effort can estimate it from the Daily 3 series
alone. This is a Part E item, not a Part D one.

**[INF]** N1/N3 inject **whole sets**; N2 injects **two whole sets**. This is structurally
important: the document's own anomaly vocabulary is *set-quantised*. Only N4 injects a partial,
unquantised amount, and only N5/N6 break the stream rather than lengthening it.

---

# PART B — WHAT THAT IMPLIES ABOUT OBSERVABLE STRUCTURE

Stated as testable propositions, per the brief. **Nothing in this Part is a conclusion.**

## B.0 Notation, and the one modelling choice that must be made explicit

**[CODE]** TFM's constant-skip kernel consumes **one generator output per observed draw** and burns
`skip` outputs between consecutive observations — `prng_registry.py:981-989`:

```c
:981  for (int i = 0; i < k; i++) {
:982      state = (a * state + c) & m;                       // consume ONE output …
:983      unsigned int output = (state >> 16) & 0xFFFFFFFF;
:984-986  if ((output % 1000) == residues[i] % 1000 && …) matches++;   // … per observed draw
:987      for (int s = 0; s < skip; s++) state = (a*state+c) & m;      // burn `skip`
```

So **stride between consecutive observations = `skip + 1`**, and one `skip` unit is one output
consumed and not published. **[DATA]** A residue is one integer per draw, `0…999`
(`daily3.json`: `{"date": "2000-01-01", "session": "evening", "draw": 390}`; all 18,068 values in
range).

The document's specs (A.2) count **selections**, not outputs. Bridging them requires a choice the
document does not make:

| hypothesis | statement | status |
|---|---|---|
| **H-A** | the RNS produces **one value per game draw**, and Daily 3's three digits are formatted from it | **the model TFM implicitly assumes** [CODE] `prng_registry.py:982-986`. **[DOC] Unsupported by the document** |
| **H-B** | the RNS makes **one selection per digit**, per the spec `03:00-09r` = 3 selections | **[INF] the reading the specs most directly support** |

Both accountings are given below. **[DOC]** The document supports neither over the other on the
question of outputs, because it never describes the machine's sampling at all.

## B.1 — P1. Between two consecutive MIDDAY Daily 3 values

**Session inventory [INF] from A.1, A.2, A.5** (no anomaly, document-effective era):

| session | sets consumed | selections (H-B) | game-draws (H-A) |
|---|---|---|---|
| midday | pre-test D3 + live D3 | 3 + 3 = **6** | 1 + 1 = **2** |
| evening | pre-test {D3,D4,F5,DD} + live {D3,D4,F5,DD} | 18 + 18 = **36** | 4 + 4 = **8** |
| **per calendar day** | | **42** | **10** |

**Arithmetic, H-A.** Consecutive midday observations are one calendar day apart. Between them lie:
the remainder of day N's midday session (0 — Daily 3 is the last and only game), day N's **entire
evening session** (8), and day N+1's midday **pre-test** (1).

```
gap  = 0 + 8 + 1 = 9 burned outputs
stride = 9 + 1 = 10 = the daily inventory
⇒ TFM skip = 9                     [P1-A]
```

**Arithmetic, H-B.** The three digits of one draw occupy three *consecutive* outputs `t, t+1, t+2`;
the next midday draw occupies `t+42, t+43, t+44`.

```
within-draw stride = 1  (skip 0);  between-draw gap = 40  (skip 39)
⇒ no single constant skip exists   [P1-B]
```

**P1.** *Under H-A and one continuous trajectory, consecutive midday Daily 3 values are separated by
exactly **9** burned outputs.* Falsified by: no seed in the searched domain achieving above-chance
match rate at `skip = 9` on a midday-scoped window, when the same geometry at other skips does no
better.

## B.2 — P2. Between two consecutive EVENING Daily 3 values

**[INF]** Let `a` = game-draws after D3 within the evening live set and `b` = those before it,
`a + b = 3` (H-A). Between two consecutive evening observations lie: `a`, then day N+1's **entire
midday session** (2), then day N+1's evening **pre-test set** (4), then `b`.

```
gap = a + 2 + 4 + b = (a+b) + 6 = 3 + 6 = 9
⇒ TFM skip = 9                     [P2-A]
```

**P2.** *The evening gap is **identical to the midday gap (9)** and is **invariant to the order of
the four evening games**, because `a + b` is fixed.* **[INF]** This is a strong and slightly
surprising result: it means the unknown in A.1 (draw order) **does not matter** for a
session-scoped series. Both series have the same period-total — the full daily inventory of 10 —
because each publishes exactly one value per day.

**Under H-B**: `gap = (a+b in selections) + 6 + 36 - … ` resolves to a **between-draw gap of 40
(skip 39)** with **skip 0 inside each draw**, again order-invariant, and again not a constant skip.

**[INF] P2 corollary — the combined chronological stream is NOT constant-skip.** If both sessions
are used in true time order, with `p` = D3's 0-based position in the evening live set:

```
midday → evening :  skip = 4 + p        (H-A)
evening → midday :  skip = 4 - p
                    the two always sum to 8; stride pair sums to 10
p ∈ {0,1,2,3}  ⇒  (4,4), (5,3), (6,2), (7,1)
```

**P2b.** *A chronologically ordered combined stream has a **period-2 alternating skip**, and its
two values identify `p` — the evening draw order the document declines to state.* This is a
falsifiable prediction with an identifiable parameter, and it is **not** what the current combined
path would see — see C.3.

## B.3 — P3. The era boundary the dataset itself creates

**[DATA]** Measured live from `daily3.json` this session:

```
first midday record anywhere:     2002-11-04   (combined index 1039)
first date carrying BOTH sessions: 2002-11-04
evening-only dates:                1,039  (last: 2019-01-25, an isolated anomaly)
```

**[INF]** For any day with **no midday session**, the daily inventory is the evening session alone:
`8` game-draws (H-A), so the evening-series stride is `8` and **`skip = 7`**, not 9.

**P3.** *If the document's session structure is projected backwards, the evening series should show
**skip 7 before 2002-11-04 and skip 9 after** — a regime change at a date known independently of
the draw values.* **[DOC] The document does not govern the pre-2021 period at all** (see B.6), so
P3's premise is an extrapolation, not a derivation. It is stated because **a search that finds a
single global constant skip across the whole series is, under any version of this accounting,
finding something the process should not produce.**

## B.4 — P4. Anomaly perturbations are set-quantised, not dense

**[INF]** From A.8: N1/N3 inject whole sets, N2 injects two. For a **midday-scoped** series under
H-A, an extra evening pre-test adds 4 and an extra midday pre-test adds 1:

```
midday-series skip alphabet  ⊆  { 9 }  ∪  { 9 + 1·i + 4·j }   for small integers i, j ≥ 0
evening-series skip alphabet ⊆  { 9 }  ∪  { 9 + 1·i + 4·j }   (same inventory, same increments)
    → {9, 10, 11, 13, 14, 17, …}
```

**P4.** *The between-draw skip takes values on a **coarse lattice generated by set sizes {1, 4}
(H-A) or {3, 18} (H-B)**, not on a dense integer range.* Falsified by: a variable-skip search whose
recovered sequences populate the lattice's gaps as densely as its points.

**[INF]** The searched range is `skip_max ≤ 250` **[CODE]** `distributed_config.json` →
`search_bounds.skip_max.max = 250`. Under H-A, `skip = 250` corresponds to roughly **25 consecutive
days of undocumented anomalies**; under H-B, to about six days. **P4 does not say 250 is wrong — it
says the range is populated far more sparsely than a uniform prior over it assumes.**

## B.5 — P5. The machine identity is a per-session coin, and it is invisible

**[DOC]** From A.3: exactly two automatic machines; §II selects one per session by an RNG program;
the identity is recorded only on retained paperwork.

**P5.** *Even a **session-scoped** series is a **mixture over two machines**, re-drawn at every
session.* **[INF]** If each machine carries its own generator state, a window of `n` consecutive
observations from one session-scoped series lies entirely within one machine's run with probability
`2^-(n-1)` under uniform independent selection — `1/32` at `n = 6` (the current minimum window),
`1/128` at `n = 8`, `1/2^49` at `n = 50` (the current maximum).

**[DOC] The document does not say the two machines carry independent generator state.** It says
nothing about generator state at all. **[SPEC]** That they do is the natural reading of two
physically separate machines each announcing *"RNS on line"* at its own power-on, but it is
speculation, and P5 is stated so that it can be tested rather than assumed.

**[INF] P5 is the proposition that would explain S112's result** — cited from the brief: real data
optimises at **W8** and *"short-lived regimes, not one continuous seed stream."* A two-machine
mixture predicts short coherent runs. It predicts them *shorter* than 8, which is itself a
discriminating test (D.3).

## B.6 — P6. The document's effective date versus the data

**[DOC]** The document is effective **June 9, 2021** and carries no statement about any earlier or
later configuration. **[DOC] It does not describe the evening game set at any other date.**

**[DATA]** Filtered-array index of the first record on or after 2021-06-09:

| session filter | filtered index of first governed record | filtered length |
|---|---|---|
| `['midday']` | **6,791** | 8,515 |
| `['evening']` | **7,830** | 9,553 |
| `['midday','evening']` | **14,621** | 18,068 |

**P6.** *Every proposition B.1–B.5 is a claim about draws on or after index ~6,791 (midday) /
~7,830 (evening). Applying them to earlier draws asserts, without evidence, that the 2021 session
structure held earlier.* Fantasy 5, Daily 4 and Daily Derby may or may not have been co-drawn in
the evening session in 2000. **[DOC] The document is silent, and this report does not guess.**

## B.7 — Fixed versus variable

| # | quantity | fixed / variable | visible in published data? |
|---|---|---|---|
| 1 | selections per D3 draw (3), per D4 (4) | **fixed** [DOC] | — |
| 2 | games per session; two sets per session (pre-test + live) | **fixed** [DOC] §V.8, §V.14, §VII.2 | — |
| 3 | **the per-day inventory, hence the session-scoped skip** | **fixed** [INF] given H-A/H-B and no anomaly | **no** — inferred, never recorded |
| 4 | draw order within the evening live set | **fixed but unstated** [DOC] | **no** — and P2 shows it does not matter |
| 5 | selections consumed by `u` games (F5, DD horses) | **variable** if rejection sampling, **fixed** if direct [DOC: unspecified] | **no** — collisions leave no trace |
| 6 | **which of two machines** [DOC] §II | **variable, per session** | **no** — retained paperwork only |
| 7 | **which RNG, A or B** [DOC] §V.4 | **unspecified whether it even varies** | **no** — recorded nowhere |
| 8 | whether an anomaly injected extra sets (N1–N4) | **variable, rate unknown** [DOC] §XXIV | **no** — anomaly reports are not published |
| 9 | whether the machine changed mid-session (N5) | **variable** [DOC] §VII.5 | **no** |
| 10 | generator state across `[Shut Down]` → power-on | **unspecified** [DOC] §VII.8, §V.2-3 | **no** |
| 11 | outputs consumed per selection | **unspecified** [DOC] | **no** |

**[INF]** Rows 6, 7, 8, 9, 10 and 11 are all **variable-or-unknown AND invisible**. Rows 1–4 are
fixed. **The document's structure divides cleanly: what is fixed is knowable, and what varies is
unrecorded.** That division is the whole content of Part E.

---

# PART C — WHERE TFM'S CURRENT MODEL MATCHES, AND WHERE IT DOES NOT

Cross-referenced against `docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md` (HEAD `8bbe79e`,
753 lines) per the brief — **cited, not re-derived**. Its established findings are taken as given:
one unbroken 48-bit LCG trajectory; gaps as `skip` burns; no machine identity, no A/B branch, no
reseed, no session/date input in the draw loop; the searched bounds; and S112's W8 result.

**Governed-status discipline (skill §1.1).** Several items below are already diagnosed and are
reported as **status, not findings**: `skip_min`/`skip_max` never reach the hybrid kernels
(skill §2.7 #4); forward hybrids ignore `offset` (§2.7 #5); `offset` is one scalar doing two jobs
(Chapter 2 F-4, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1133`); `skip_learning_rate` dead
(§2.7 #6); combined-session still sampler-reachable (Chapter 1 §8.3.1); **and the analysis window
sitting on the oldest records** (`docs/DAILY3_CONSUMER_CONTRACT_v1.md:198-212`).

## C.1 Element-by-element

| Part-B element | current model | verdict | does it matter? |
|---|---|---|---|
| **unpublished pre-test set** (A.5) | **REPRESENTED** — this is precisely what `skip` was built for; `ca_d3_threshold_calibration.py:28-35` names *"one pre-test cycle"* explicitly | ✅ | No — represented, and now anchored |
| **co-drawn evening games** (A.1) | **REPRESENTED** — same `skip` abstraction | ✅ | No |
| **the skip MAGNITUDE (9)** (P1, P2) | **ABSTRACTED into a searched range.** `skip_min ∈ [0,10]`, `skip_max ∈ [10,250]` **[CODE]** `distributed_config.json`; the kernel sweeps every value and keeps the argmax **[CODE]** `prng_registry.py:972`, `:990-994` | ⚠️ | **Harmless-to-favourable.** 9 is inside the swept range for any trial with `skip_min ≤ 9`. The cost is dilution, not exclusion — see C.2 |
| **order-invariance of the session gap** (P2) | **not represented, not needed** | ✅ | No — P2 shows the unknown cancels |
| **the skip LATTICE** (P4) | **IGNORED.** Constant mode sweeps a dense integer range; hybrid mode does a greedy per-draw adaptive search re-centred on each hit, from a hardcoded `expected_skip = 5` **[CODE]** `prng_registry.py:1027`, `:1033-1049` | ⚠️ | **Matters for selectivity, not reachability** — C.2 |
| **per-session equipment re-selection** (A.3, P5) | **IGNORED.** No machine identity anywhere (continuity report Q2a: *"NO EVIDENCE FOUND"*) | ❌ | **Potentially fatal — C.4** |
| **A/B RNG per machine** (A.4) | **IGNORED** (continuity report Q2b, Observation 3) | ❌ | Same class as C.4, one level finer |
| **power cycle per session** (A.6) | **IGNORED.** The kernel carries one state across every observation; continuity report Observation 4: *"Skip can represent outputs consumed; it cannot represent state discarded"* | ❌ | **Potentially fatal — C.4** |
| **anomaly-injected sets** (A.8) | **ABSTRACTED** — absorbed into a wider skip in constant mode; in hybrid mode absorbed by tolerance, `skip_tolerance ∈ {5,20,10,50}` **[CODE]** `hybrid_strategy.py:37,46,55,64,73` | ✅ | No — this is exactly what loose, variable skip is for |
| **`u`-game selection variance** (A.2) | **ABSTRACTED** into skip (constant mode cannot vary per gap; hybrid can) | ⚠️ | Minor under H-A; under H-B it is subsumed by C.5 |
| **selections per draw = 3** (A.2) | **CONTRADICTED.** The kernel tests **one output** against the **full three-digit value** mod 1000 **[CODE]** `prng_registry.py:982-986` | ❌ | **Decisive if H-B is true — C.5** |
| **the document's effective date** (P6) | **IGNORED, and structurally unreachable** — C.3 | ❌ | **Decisive for testing the document at all** |

## C.2 Where an unmodelled detail is harmless — selectivity carries the weight

The owner's framing is that the sieve is a **candidate filter**, not a state-recovery attack: the
survivor is evidence that an alignment resolves, and the ML learns from *how* it survived. Three of
the mismatches above are genuinely absorbed by that framing, and it is worth being precise about
why.

**[INF]** The bidirectional test's selectivity is the reason. Per whitepaper §0.2 as summarised in
skill §0.2, forward and reverse survival are approximately independent for incorrect seeds, so
`P(survive both) ≈ P(survive forward)²` — the exponent squares. A skip value that is *wrong but
searched* costs **dilution of the survivor pool**, not correctness: the kernel takes the argmax over
the swept range **[CODE]** `prng_registry.py:991-995`, so a correct alignment at `skip = 9` still
wins its own sweep. Extra swept values raise the false-survival rate by a factor of at most the
number of hypotheses tried, which is a linear penalty against an exponential filter.

**Therefore these are harmless:**
- **Not knowing the magnitude is 9** — it is inside the range. The penalty is up to ~251×
  false-positive inflation against an exponential discriminant, and it costs proportional kernel
  work (D.2 quantifies the recoverable waste).
- **Anomaly-injected sets** — a wider skip or a hybrid tolerance absorbs them. This is the
  designed-for case.
- **`u`-game selection variance** — same class.

## C.3 Where the search cannot reach the data the document governs — **the finding**

**[CODE]** `offset` is bounded `[0, 100]` (`distributed_config.json` → `search_bounds.offset`;
`window_optimizer.py:142-143`), `window_size` is bounded `[6, 50]`
(`search_bounds.window_size`), and `offset` slices the residue array **from the oldest end**
**[CODE]** `miner/range_miner_worker.py:648-649`:

```python
:648   start = max(0, min(int(offset), n - window_size))
:649   window = data[start:start + window_size]
```

```
maximum filtered index the production sieve can reach  =  100 + 50  =  149
```

**[DATA]** Against the measured index of the first document-governed record (P6):

| session filter | max reachable index | first governed index | reachable dates | governed dates |
|---|---|---|---|---|
| `['midday']` | 149 | **6,791** | 2002-11-04 … 2003-04-02 | 2021-06-09 … |
| `['evening']` | 149 | **7,830** | 2000-01-01 … 2000-05-29 | 2021-06-09 … |
| both | 149 | **14,621** | 2000-01-01 … 2000-05-29 | 2021-06-09 … |

> **THE PRODUCTION SIEVE CANNOT EXAMINE A SINGLE DRAW THAT THIS DOCUMENT GOVERNS.**
> The reachable window ends 18 years and roughly 6,600 records short of the effective date.

**Governed status, not a new finding.** `docs/DAILY3_CONSUMER_CONTRACT_v1.md:198-212` already
records the mechanism and its consequence in as many words — *"**The production sieve analyses draws
from March 2000**"* — using the live `optimal_window_config.json` (`window_size: 21`, `offset: 66`)
as its worked example. Chapter 1 (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:369-371`) records that
`offset` slices from the oldest end. **What is new here is only the collision with the document's
effective date**, which could not be stated before the document was available.

**[INF] Consequence for this brief.** Any approach that claims to test the procedures document
must first move the analysis window. A null result from the current geometry is **not evidence
against the document's structure** — it is a result about draws the document does not describe.

## C.4 Where an unmodelled detail could make the search mathematically unable to succeed

Three of the ignored elements are not absorbed by selectivity, because they are not *consumption* —
they are **stream identity**. `skip` can represent an output consumed; it cannot represent a
different trajectory. The continuity report states this exactly (Observation 4).

| element | why selectivity cannot rescue it |
|---|---|
| **per-session machine re-selection** (P5) | Under two independent machine states, a window of `n` observations from one session series lies within one trajectory with probability `2^-(n-1)`. At the **minimum** searched window `n = 6` that is `1/32`; at `n = 50` it is `2^-49`. **No threshold, no bidirectionality and no seed count recovers a window that was never generated by one trajectory.** |
| **power cycle per session** (A.6) | If state does not survive `[Shut Down]`, **every** window spanning a session boundary is cross-trajectory — which is every window in a session-scoped series with `n ≥ 2`, since consecutive observations in either series are one day apart. |
| **A/B RNG** (A.4) | Same shape, one level finer, and unquantifiable — the document does not even say whether A/B varies per value. |

**[INF] The honest statement of the boundary the brief asks for:** an unmodelled physical detail is
harmless when it changes **how many outputs were consumed on one trajectory** — skip absorbs it and
selectivity pays a linear price. It is fatal when it changes **which trajectory produced the
value** — no parameter in the current search space can express that, so the search is not
mis-tuned, it is searching for an object the process may not produce.

**[INF]** Crucially, this is **not** an argument that the sieve is worthless. Under the
candidate-filter framing, a two-machine mixture predicts that **some** windows are single-trajectory
and most are not, at a computable rate. That converts C.4 from a fatal objection into **D.3's
falsifiable test** — and it makes the *distribution* of successful window positions the signal,
rather than the existence of a global fit.

## C.5 The residue-width mismatch — the one place the model may be searching for nothing

**[CODE]** The kernel's per-draw test is one output against the full three-digit value:

```c
:982  state = (a * state + c) & m;
:983  unsigned int output = (state >> 16) & 0xFFFFFFFF;
:984  if (((output % 1000) == (unsigned int)(residues[i] % 1000)) && …
```

**[DOC]** The document's Daily 3 spec is `03:00-09r` — **three selections from 0–9**.

**[INF]** If **H-B** holds (each digit is its own selection), then no single generator output equals
the published three-digit value under any correct model of the machine, and **the current kernel's
test can never be satisfied by the true generator, at any seed, any skip, any window, any
threshold.** This is the strict sense of "mathematically unable to succeed" the brief asks about,
and it is the only element in Part B that has that property.

**[INF]** If **H-A** holds (the machine draws one value and formats digits), the current test is
correct in form. **[DOC] The document does not decide between them**, and neither does anything in
the repository — the three-lane CRT test is exactly equivalent to `% 1000`
(`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6; skill §2.17b), so no existing lane carries digit-level
information.

**[CODE]** The existing residue-source seam is `entry.get("full_state", entry["draw"])`
(`miner/range_miner_worker.py:650`), Chapter 2 **F-8** — and Chapter 2 is explicit that it
*"changes the **residue source, not the comparison width**: the predicate still reduces mod
1000/8/125"* (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1137`). **So H-B is not reachable through the
existing hook.** Testing it requires new kernel semantics — D.4.

## C.6 What this document settles in TFM's favour

For balance, three places where the document **confirms** current practice rather than challenging
it:

1. **`skip` is physically real and now properly cited.** A.5 supplies §V.14 / §V.16 Note / §XI.16
   Note / §XII.1 for skill §0.4's `UNAVAILABLE` citation, and A.1 supplies the co-drawn-games
   premise. The near-removal of `skip_min`/`skip_max` recorded in skill §0 was, on the document's
   evidence, correctly stopped.
2. **Per-session scoping rests on a true premise** (A.3.1) — the ruling chain in skill §2.10b is
   sound as far as it goes, and the *"combined-session sequential sieve prohibited by default"*
   ruling is **strengthened**, not weakened, by P2b: the stored container order is
   anti-chronological (C.7), so the combined stream is not merely mixed but time-reversed within
   each date.
3. **Loose thresholds remain necessary** (skill §0.3) — nothing in the document bears on this, and
   nothing here should be read as arguing for tightening.

## C.7 One live measurement that bears on C.6.2

**[DATA]** Measured this session over all 18,068 records: **every one of the 8,514 same-date
adjacent pairs is `(evening, midday)`** — evening precedes midday in storage, while midday
(live by 1:10pm) precedes evening (live by 6:40pm) in time.

**[CODE]** `load_residue_window` preserves stored order and never re-sorts
(`miner/range_miner_worker.py:641-650`). **[INF]** A `sessions=['midday','evening']` window is
therefore **time-reversed within each date**, which is a stronger reason to prohibit it than
mixing alone. **Governed status:** `docs/DAILY3_CONSUMER_CONTRACT_v1.md:418` already records
assumption 3 — *"Intra-day sort key is `(date, session)` raw (evening first)"* — and skill §2.14
records the canonical order. **Reported as status.**

---

# PART D — THE PLAN OF ATTACK

Four approaches, ranked by expected information per unit of cluster cost. **D.2 is the one
executable within current governed constraints** (2^32 domain, existing bounds, no new production
code), as the brief requires.

**Cost units.** **[CODE]** `agent_manifests/window_optimizer.json` → `miner_stripe_size =
67,108,864` (2^26 seeds), `miner_substripes = 8` (2^23 per sub-stripe), `max_seeds =
1,073,741,824` (2^30 — note this is **one quarter** of the governed 2^32 domain).

```
full governed domain [0, 2^32)  =   64 stripes  =  512 sub-stripes   per direction, per trial
manifest default (2^30)         =   16 stripes  =  128 sub-stripes
```

**No wall-clock or seeds/sec figure appears anywhere below.** Per the brief and the standing
finding, `elapsed_s` is computed by workers and dropped at the ledger boundary; there is no
trustworthy throughput baseline. Where an approach is cheaper, that is stated as **kernel work per
seed**, which is directly countable from the kernel source.

---

## D.1 — RANK 1: Move the analysis window onto draws the document governs

**Hypothesis (falsifiable).** *The procedures document describes the process that produced draws on
and after 2021-06-09. Propositions P1/P2 (`skip = 9`, session-scoped) hold on that data and are not
expected to hold on 2000–2003 data.*

**Why rank 1.** C.3 shows the current geometry cannot reach a single governed draw. **Until this
changes, no approach in this report is actually testing the document** — including D.2. This is the
prerequisite, not a preference.

**Consumes.** Data already held — `daily3.json`, 18,068 records **[DATA]**. No new data. No new
scrape (6-P2 is with Beta and is not required: the governed era is already in the file, ending
2026-02-26).

**Search geometry.** Session-scoped (`['midday']` **or** `['evening']`, never both — A.3.1, C.7).
Window `[6, 50]` unchanged. Skip as in D.2. Seed domain `[0, 2^32)` unchanged. **The one thing that
must change is `offset`'s reachable range**, from `[0, 100]` to something that spans index ~6,791+.

**Positive result.** Above-chance bidirectional survival on governed-era windows at
`skip = 9`, absent or weaker on 2000-2003 windows under identical geometry. That is a **differential**
result and is far stronger than any single-window fit, because the geometry is held constant and
only the era varies.

**Null result rules out.** That the document's session inventory, as read in Part B, produces a
single-trajectory constant-stride signal detectable by this sieve on the data the document governs.
It would **not** rule out the document's physics — C.4 and C.5 are both live alternative
explanations for a null.

**Cost.** Identical per trial to any current Step-1 trial: 64 stripes / 512 sub-stripes per
direction for the full domain. The differential design **doubles** the trial count (governed era +
control era). No new hardware, no change to fleet shape.

**What would have to change (described only, not implemented).**
1. `distributed_config.json` → `search_bounds.offset.max`, currently `100`. **This is a config
   value, not code** — but it is a **governed bound**, so it is Beta's to move, not Alpha's, and it
   is the reason this approach is not the "executable within current constraints" one.
2. **The `offset` dual-use problem must be confronted first.** Chapter 2 **F-4**
   (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1133`, **CONFIRMED, not repaired**): one scalar drives
   both the host residue slice and the device pre-advance, *"coherent only at `skip = 0`"*. Raising
   `max_offset` to ~7,000 would make the constant kernel pre-advance the generator by up to 7,000
   steps **[CODE]** `prng_registry.py:974-976` as a side effect of choosing which draws to look at —
   silently changing the meaning of every trial. **Chapter 2 rules that F-4 belongs in the future
   hybrid input-semantics design, not a standalone arithmetic patch.** So the honest statement is:
   *this approach needs a separate window-anchor input, distinct from the generator phase* — which
   is a design, not a bound change.
3. Forward hybrid kernels take no `offset` at all (skill §2.7 #5, **OPEN**), so on that path the
   window would shift with no pre-advance — a third semantics.

**Blocking dependency, stated plainly:** D.1 cannot be executed by moving one number. It needs the
window-anchor/generator-phase separation that F-4 already names. **That is a proposal to Beta, and
it is the highest-value one in this report.**

---

## D.2 — RANK 2: Skip-pinned, session-scoped constant sieve — **EXECUTABLE NOW**

**Hypothesis (falsifiable).** *Within one session-scoped stream, consecutive Daily 3 values are
separated by a constant `skip = 9` (P1, P2) on a single 48-bit LCG trajectory in `[0, 2^32)`.*

**Why this is the runnable one.** It requires **no new production code, no bound change, no config
outside existing ranges, and no new data.** `skip = 9` is inside the existing search space: the
tightest reachable pinning is `skip_min = 9`, `skip_max = 10` **[CODE]** — `skip_min` is bounded
`[0,10]` and `skip_max` is sampled from `max(skip_min, 10)` upward
(`window_optimizer_bayesian.py:538-543`), so a two-value sweep `{9, 10}` is the minimum, and it
brackets the prediction.

**Consumes.** Data already held. No new data.

**Search geometry.**
```
sessions      ['midday']  and, separately, ['evening']     — never both (A.3.1, C.7)
skip_min      9        skip_max      10                    — 2 swept values, vs up to 251 today
window_size   [6, 50]   unchanged
offset        [0, 100]  unchanged  ⇒ midday: 2002-11-04…2003-04-02 ; evening: 2000-01-01…2000-05-29
seed domain   [0, 2^32) unchanged                          — Seed-Domain v1.1
thresholds    unchanged, Optuna-tuned per direction        — skill §0.3
```

**Positive result.** Bidirectional survivors concentrating at `best_skip = 9`
**[CODE]** `best_skips[pos]` is recorded per survivor (`prng_registry.py:1001`, `unsigned char`, so 9
stores exactly), at a rate materially above the same geometry with skip pinned to a control value
(e.g. `{17,18}`) on the same window. **A matched control arm is mandatory** — without it the result
is uninterpretable, because a two-value sweep changes the false-positive rate as well as the
hypothesis.

**Null result rules out.** Very little, and this must be stated up front so a null is not
over-read:
- **P6/C.3**: the reachable window is 2000–2003, which the document **does not govern**. A null is
  consistent with the document being entirely correct about 2021+.
- **P3**: for the evening filter the reachable window is the **evening-only era**, where the
  derived value is **7**, not 9. *(An evening-filter arm should therefore pin `{7,8}` — but note
  `skip_max` cannot go below 10 **[CODE]** `search_bounds.skip_max.min = 10`, so the tightest
  evening pinning containing 7 is `skip_min = 7, skip_max = 10`, four values.)*
- **C.4/C.5** remain live alternative explanations for any null.

**What a null DOES rule out:** that a constant stride of 9 (or 7) with a single `[0,2^32)`-domain
48-bit LCG trajectory fits the earliest retained Daily 3 records. That is a narrow but real result,
and it is currently **untested** — the production configuration sweeps skip broadly and has never
isolated the document-derived value.

**Cost — and this approach is CHEAPER than the status quo.** The constant kernel's outer loop runs
once per swept skip value **[CODE]** `prng_registry.py:972`:
```c
:972  for (int skip = skip_min; skip <= skip_max; skip++) {   // full re-scan of the window per value
```
```
Optuna's widest current sample   skip_min=0,  skip_max=250   →  251 passes per seed
D.2 midday arm                   skip_min=9,  skip_max=10    →    2 passes per seed
                                                              ⇒ ~125× less kernel work per seed
```
At identical stripe count — 64 stripes / 512 sub-stripes per direction for the full 2^32 domain —
D.2 does a small fraction of the per-seed work of a wide-skip trial. **This is a countable
statement from the kernel source, not a throughput estimate.**

**What would have to change (described only).** **Nothing in production code.** The values are
already inside the governed bounds and inside the sampler's range. Reaching them requires one of:
`study.enqueue_trial` with fixed parameters — the existing S166 warm-start path **[CODE]**
`window_optimizer_bayesian.py:786`, whose enqueued parameter dict **already carries exactly the
keys this approach needs** (`window_optimizer_bayesian.py:782-784`: `'skip_min'`, `'skip_max'`,
`'forward_threshold'`, `'reverse_threshold'`, `'session_idx'`); or a narrowed `search_bounds` in
`distributed_config.json` (a config change, within existing bounds).

> *(Anchor note: skill §2.7 #4 cites `prng_registry.py:1047` for the hardcoded `expected_skip`;
> live at `8bbe79e` the `java_lcg` hybrid's is `:1027`, and the skill's warm-start anchor `:725`
> is live at `:786`. Line drift only — the mechanisms are unchanged and both were re-read this
> session.)*

> **⚠ Route caution — skill §2.15.** `skip_min` and `skip_max` are **not** in
> `agent_manifests/window_optimizer.json` → `default_params` **[CODE]** (verified live this
> session; the declared keys are `enable_pruning, lottery_file, max_seeds, min_workers,
> miner_output_dir, miner_stripe_size, miner_substripes, n_parallel, output, prng_type,
> pwc_transport, resume_checkpoint, resume_study, seed_cap_amd, seed_cap_nvidia, seed_start,
> staging_*, staging_dir, strategy, study_name, test_both_modes, trse_context,
> use_persistent_workers, use_range_miner, use_zmq_sqlite, window_trials, worker_pool_size`).
> **WATCHER's step-scoped filter drops any key not declared** — `agents/watcher_agent.py:1290-1314`.
> So a WATCHER-driven run **cannot** carry a pinned skip today; the `search_bounds` route in
> `distributed_config.json` is the one that reaches the sampler. **Gate the route, not the
> parameter.**

**Standing constraint.** Any execution is **MICHAEL-INITIATED ONLY** and is subject to the Beta
holds recorded in skill §8 (gate 12 and the Phase-7 soak are both HELD). Nothing here requests or
implies authorization.

---

## D.3 — RANK 3: Test the two-machine mixture directly (P5)

**Hypothesis (falsifiable).** *Because equipment is re-selected every session from exactly two
machines (A.3), a session-scoped series is a mixture of two trajectories. Survivor yield should
therefore fall off approximately as `2^-(n-1)` in window length `n`, and successful window start
positions should be **clustered**, not uniformly distributed.*

**Why rank 3.** Its **first stage costs no cluster time at all** — it is analysis of artifacts
already on disk — and it is the only approach that offers a positive explanation for S112's W8
result rather than treating it as noise.

**Consumes.**
- *Stage 1 (free):* existing survivor NPZs and Optuna study rows already held. **[CODE]** The
  22-array contract carries `seeds`, `forward_matches`, `reverse_matches`, `score` as the four
  per-seed columns (skill §2.3); `window_size` and `offset` are recorded per trial.
- *Stage 2:* a window-length sweep at fixed skip — existing code, existing bounds.

**Search geometry.** `window_size` swept across its full `[6, 50]` at otherwise fixed geometry,
session-scoped, `[0, 2^32)`. Stage 1 needs no run at all.

**Positive result.** A yield-vs-`n` curve with an **exponential** shape and a decay constant near
`1/2` per additional draw, plus non-uniform clustering of successful `offset` positions. **[INF]**
Note this predicts the optimum at the *smallest* admissible window, which sits in tension with
S112's W8 and consistent with the W2/W3 result on the PA dataset recorded in
`docs/SESSION_CHANGELOG_20260314_S143.md:120-124` (cited via the continuity report `:598-604`) —
that tension is itself the discriminating measurement.

**Null result rules out.** A uniform-random per-session choice between two independent-state
machines. It would **not** rule out two machines sharing state, or the RNG program selecting with
memory, or one machine dominating in practice.

**Cost.** Stage 1: **zero cluster cost** — read-only analysis of held artifacts.
Stage 2: one full-domain pass per window-length point, 64 stripes / 512 sub-stripes per direction
each. A 6-point sweep is 6× a standard trial.

**What would have to change (described only).** Stage 1: nothing — a read-only analysis script
outside production. Stage 2: nothing in production code; `window_size` is already a sampled
dimension **[CODE]** `window_optimizer_bayesian.py:529-531`. **Caveat:** the `offset` dual-use
problem (C.3, F-4) means sweeping `offset` to vary window position also varies generator phase, so
Stage 2's clustering measurement is confounded until D.1's separation exists. **Stage 1 is not
confounded** and should be done first.

---

## D.4 — RANK 4: Per-selection residue semantics (H-B)

**Hypothesis (falsifiable).** *Each Daily 3 digit is a separate machine selection (spec
`03:00-09r`), so a draw corresponds to **three consecutive generator outputs matched mod 10**, not
one output matched mod 1000.*

**Why rank 4 despite the highest stakes.** If true, **no current or proposed run in this report can
ever succeed** (C.5) — it is the only hypothesis whose truth invalidates all the others. But it is
also the most expensive to test and the furthest from the current implementation, and **the
document does not favour it over H-A on the output question** — it constrains *selections*, and is
silent on outputs.

**Consumes.** Data already held. The residue array would carry the same 18,068 values, decomposed
into digit triples at the host.

**Search geometry.**
```
per draw      3 consecutive outputs, each tested (output % 10) == digit
inter-draw    skip = 39  (H-B, P1-B)   — inside skip_max ≤ 250, so the RANGE is reachable
intra-draw    skip = 0                 — requires a per-position skip pattern, not a constant
window        [6, 50] draws  =  [18, 150] outputs
seed domain   [0, 2^32) unchanged
```

**[INF]** Note the **filter strength is identical** — `1/1000` per draw either way — so this is not
a weaker test. It is a **different alignment of the same amount of evidence**, and it advances the
state 3× faster per draw.

**Positive result.** Survivors under digit-triple semantics where none exist under mod-1000
semantics, on the same window and seed domain. This would be an unambiguous, structural result.

**Null result rules out.** H-B, given the rest of the model (single trajectory, `[0,2^32)`, java_lcg,
`skip = 39`). Because that conjunction is long, a null here is weak evidence — the standard caution
for any compound hypothesis.

**Cost.** Comparable per-seed work to a pinned-skip constant trial: 3 output-consumptions plus 39
burns per draw, versus 1 plus 9. Roughly **4× the generator steps per seed per window**, at the same
64 stripes / 512 sub-stripes per direction. Not cheap, not prohibitive.

**What would have to change (described only).**
1. **A new kernel variant** with a mod-10 comparison lane and a fixed intra-draw stride of 1. The
   existing `full_state` hook is **not** the route — Chapter 2 **F-8**
   (`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1137`) states it changes the residue **source, not the
   comparison width**, and *"the predicate still reduces mod 1000/8/125"*.
2. **A host-side residue expansion** producing `3k` digit residues from `k` draws, which changes
   `residue_sha256` and therefore every assignment payload — the shared residue authority is
   `load_residue_window` **[CODE]** `miner/range_miner_worker.py:602-650`, and the D6 correction
   note there is explicit that **exactly one function may own that derivation**. Any new semantics
   must go **through** it, not beside it.
3. **The 22-array NPZ contract is frozen** (skill §2.3, §4) — a digit-level sieve must still emit
   exactly 22 arrays with per-draw survivor semantics, or it is a different contract requiring
   governance.
4. **Reserved authority.** "Sieve strategy/mathematics" and "feature engineering" are human-only per
   skill §2.13. **This is a Beta proposal, not an Alpha implementation, and this report does not
   propose building it — only that the question be decided.**

**[INF] Cheapest possible discriminator, offered as an alternative to building anything:** H-A and
H-B differ in how many generator outputs a day consumes (10 vs 42). If D.2 or D.1 ever produces a
confirmed survivor, its recorded `best_skip` **[CODE]** `prng_registry.py:996` distinguishes them
directly — 9 supports H-A, and nothing near 9 supports H-B. **The question may not need a dedicated
run at all.**

---

## D.5 Ranking summary

| rank | approach | executable now? | cluster cost | kills the others if true? |
|---|---|---|---|---|
| **1** | **D.1** move the window onto governed draws | **No** — needs window-anchor / generator-phase separation (F-4) | 2× standard trial (differential) | No, but every other result is uninterpretable without it |
| **2** | **D.2** skip-pinned session-scoped sieve | **YES — within existing bounds, no new code** | **~125× less kernel work per seed** than a wide-skip trial, same stripe count | No |
| **3** | **D.3** two-machine mixture test | **Stage 1 YES, free** | Stage 1 zero; Stage 2 ~6× standard trial | Partially — a confirmed mixture bounds every window length |
| **4** | **D.4** per-selection residue semantics | No — new kernel + residue expansion | ~4× generator steps per seed | **Yes — if true, D.1–D.3 cannot succeed** |

---

# PART E — WHAT THE PUBLISHED DAILY 3 SERIES CAN NEVER REVEAL

Each item is a **[DOC]** fact about what the document routes to retained paperwork rather than
publication, plus the **[INF]** consequence. **No approach in Part D depends on any of these, and
no future approach should be proposed that does.**

| # | unavailable | document basis | consequence |
|---|---|---|---|
| **E1** | **Which of the two automatic machines produced a value.** | Selected per session by an RNG program (§II, p.4); recorded on the Draw Games Certification (§II.5) and the Official Drawing Record (§VII.7 Note, p.9); **distributed to the auditor and Internal Audits (§VII.11), never to the public path (§VIII, §IX, §X).** | **[INF]** No method can partition the series by machine from published data. A machine-aware model must either infer the partition (D.3, `2^n` hypotheses) or do without it. |
| **E2** | **Which RNG — A or B — produced a value.** | §V.4 (p.6) requires both green and says nothing else. **[DOC] Not recorded on any artifact the document names, public or retained.** | **[INF]** Strictly unavailable — not even a records request would recover it, because the document does not describe it being recorded at all. |
| **E3** | **Every pre-test value.** | Generated (§V.14), verified against the video (§V.15), documented on the Draw Games Certification for SLP (§XVI.14) — **never in the publication steps.** | **[INF]** The single largest unobserved consumer. This is exactly what `skip` abstracts, and it can only ever be abstracted, never observed. |
| **E4** | **Invalid live draws.** | §V.16 Note (p.7): *"let the invalid live draw proceed, print the draw results … noting, 'Not a valid draw – pools not yet closed'"*. Retained with the DC's documentation. | **[INF]** A whole extra live set can enter the stream and leave no public trace. |
| **E5** | **Whether any given session was perturbed at all**, and how often. | Anomaly paths N1–N6 (A.8); §XXIV (p.22): *"No specific course of action can be prescribed."* Reports are e-mailed to designated recipients. | **[INF] The perturbation rate is unobservable from the series.** It cannot be estimated, only assumed — which is a reason to prefer methods robust to it (variable skip) over methods that need it (any exact reconstruction). |
| **E6** | **Whether a `u` game consumed extra selections.** | `05:01-39u`, `03:01-12u` (§V.12); the sampling method is unspecified. | **[INF]** A rejected duplicate is invisible in the final published set by construction. Even publishing all of Fantasy 5 does not reveal it. |
| **E7** | **What the generator does at power-on or `[Shut Down]`.** | §V.2-3, §VII.8. **[DOC] The document says nothing.** | **[INF]** Whether state persists across the twice-daily power cycle is **not decidable from the document**, and from the data only indirectly — as a model comparison, never as an observation. |
| **E8** | **The generator's algorithm or vendor.** | Only *"RNS on line"* (§V.3). IGT is the online gaming system (§IV); "Criterion" names the manual ball machines (§XXI.5). | **[INF]** TFM's `java_lcg` choice has **no support in this document**, and no PRNG family can be confirmed or excluded from it. |
| **E9** | **The order of the four evening games.** | Not stated (A.1). | **[INF]** Harmless for session-scoped work — **P2 proves the session gap is order-invariant.** It matters only for the combined chronological stream (P2b), where it is the identifiable parameter `p`. |
| **E10** | **Anything about draws before 2021-06-09.** | The document carries one effective date and describes no other configuration. | **[INF]** **[DATA]** ~14,600 of the 18,068 records predate it. **The document licenses claims about roughly 19% of the dataset**, and — per C.3 — the production sieve can currently reach none of that 19%. |

---

# VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every [DOC] quotation was extracted this session from the named PDF
  (`sha256 7048b255…792f74`) via `pdftotext -layout` and carries §/page. Every [CODE] claim carries
  a `file:line` read live on VM 101 at `8bbe79e`. Every [DATA] figure was computed this session
  against the live gitignored `daily3.json`.
- **clean control:** the PDF's own page footers ("Page 1 of 22" … "Page 22 of 22") confirm complete
  extraction; the `pdfinfo` page count (23, incl. cover) matches. Section-numbering defects were
  detected by reading rather than assumed absent, and are declared above.
- **fault-injection control:** **not applicable** — this is a read-and-propose task with no
  detector to falsify. No gate was written, executed or claimed.
- **completion sentinel:** all 23 pages read; the final section (`XXVII. EMERGENCY OFF-SITE DRAWS`,
  p.22) and the closing footer were reached.
- **unavailable-observer behavior:** Part E enumerates what the document routes away from
  publication. **[DOC] Silence is reported as silence** — "the document does not specify" is used
  literally throughout and never converted into a negative claim about the machine.
- **audit claim scope:** claims about the procedures document, about the current Step-1 search
  geometry, and about `daily3.json`. **No claim is made about the operator's actual equipment,
  about draws after 2026-02-26, or about any PRNG family's suitability.**
- **searched surfaces:** the PDF in full · the live VM 101 tree at `8bbe79e` (`prng_registry.py`,
  `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`,
  `miner/range_miner_worker.py`, `hybrid_strategy.py`, `distributed_config.json`,
  `agent_manifests/window_optimizer.json`) · the **live gitignored dataset** `daily3.json`
  (18,068 records, invisible to any clone-scoped audit) · `git ls-files`, `git check-ignore`,
  `git log --all -- docs/reference/`.
- **governance trail searched (`TB_RULING*`, `PROPOSAL*`, `TEAM_ALPHA*`):** yes —
  `docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md` (read in full), the
  `TEAM_ALPHA_PUSHBACK_ORDERING_AND_THRESHOLD_REGRESSION.md` gloss it quotes, and skill §2.10b's
  session-stream rulings. **Per skill §1.1 (VIR-6 addendum), `docs/` was a mandatory surface and
  was searched.**
- **chapters searched:** `CHAPTER_1_WINDOW_OPTIMIZER.md` (§3.1-3.2, F-4 treatment),
  `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (§5.1, §6, §7.3, §12 findings table F-4/F-8),
  `DAILY3_CONSUMER_CONTRACT_v1.md` (§4.3 and the assumptions table).
- **unavailable surfaces:** ser8 pre-repository archives · rig-deployed source (not compared
  against VM 101 this session) · runtime survivor NPZs and Optuna study rows (**not opened** — D.3
  Stage 1 is proposed, not performed) · the `OPERATIONAL RECOVERY SITE DRAW PROCEDURES` referenced
  by §XXVII but **not included in this PDF** · Beta ruling texts external to the tree · the bodies
  of the 40 non-`java_lcg` kernels.

---

# WHAT THIS REPORT IS NOT

- **Not an authorization or a request for one.** Nothing was launched. Gate 12 and the Phase-7 soak
  are HELD by Beta (skill §8); D.2 is described as executable, **not** as scheduled.
- **Not an implementation.** No production code written, no configuration changed, no commit.
- **Not a claim that any current component should be removed, demoted or simplified.** Per skill
  §0.4's standing rule: `skip`, loose thresholds and bidirectionality are all **confirmed** by this
  document or explicitly out of its scope. C.6 states the three places it supports current practice.
- **Not a claim that the sieve cannot work.** C.4's fatal cases are **conditional on P5/A.6 being
  true**, which the document does not establish — and D.3 converts that conditional into a
  measurement.
- **Not a re-derivation of the continuity model.** `docs/CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md`
  is cited as established throughout, per the brief.

# THE ONE-LINE ANSWER TO THE BRIEF'S QUESTION

**The best attack is the one TFM already runs — a session-scoped, loose-threshold, bidirectional
candidate filter — pointed at draws the document actually governs, with the skip magnitude the
document implies (9) tested against a control rather than diluted across a 251-value sweep.** What
differs from today is not the method but **where it looks** (C.3: the search cannot currently reach
a single governed draw) and **what it assumes about stream identity** (C.4: `skip` models
consumption and cannot model a machine change, a power cycle, or an A/B branch — and the document
mandates the first two twice a day).
