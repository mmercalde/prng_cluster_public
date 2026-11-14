# CLAUDE CODE INSTRUCTIONS — PWC AS A WORKING BACKUP, STAGE A (read-only)

## Prior record

**Searches run (public mirror, HEAD `b1cd2fa`, 2026-11-14):**

- `grep -rnIiE "PWC[^|]{0,120}(non-certifying|retired|not the authoritative|diagnostic path)|(non-certifying|retired)[^|]{0,120}PWC" docs/*.md`
- `grep -lIiE "PWC.*(retire|non-certifying|certifying authority|comparator)" docs/TB_RULING_*.md` (no hit: the ruling itself was **not found** by this search).
- `ls docs | grep -iE "PWC|QUARANTINE|RIG_CRASH|FORENSIC"`.
- `grep -n "seed_start\|start_seed\|base_seed" persistent_worker_coordinator.py zmq_sqlite_coordinator.py`.
- Read: `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md` Q1; `_R2_` `:78`.
- **Not done by Alpha:** a full search of `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`,
  `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`,
  `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/` and the
  session changelogs for PWC. **That search is this session's first task (§1, P1).**

**Documents found so far:**

| Document | What it says | Label |
|---|---|---|
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1127`, `:1420`, `:2057` | PWC hybrid (variable-skip) path is quarantined: it raises, with no override flag. PWC constant-skip still runs, as a non-certifying diagnostic comparator | ALREADY RECORDED |
| `docs/CHAPTER_12_WATCHER_AGENT.md:91` | RANGE-MINER is the certifying Step 1 engine; PWC and ZMQ are non-certifying | ALREADY RECORDED |
| `docs/CHAPTER_2_SOURCE_MAP_v1.md:363`, `:441` | The quarantine reason is named `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; Beta retired PWC from certifying authority | ALREADY RECORDED. The ruling text itself not yet located |
| `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`, `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`, `docs/RIG_CRASH_FORENSIC_20260822.md` | Listed by name. Not read by Alpha today | To be read by this session |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md` Q1 | PWC reports `failed_chunks` at pass level (`persistent_worker_coordinator.py:1508-1548`) but its trial builder does not carry that accounting (`:1725-1727`, `:1766-1768`, `:1881-1898`). So PWC earns no credit in stage 1. "Not a request to lift" the quarantine | ALREADY RECORDED |
| `persistent_worker_coordinator.py:1276` | PWC builds its chunks from a fixed 0. A non-zero `seed_start` is not honoured | ALREADY RECORDED (stage 1 report) |
| The manifest default is `use_range_miner False` (`agent_manifests/window_optimizer.json`) | A Step 1 launch without the flag takes an older route | ALREADY RECORDED |

**Items this brief raises:**

| Item | Label |
|---|---|
| **Owner, 2026-11-14:** RANGE-MINER was designed as a separate callable program with PWC and ZMQ as fallbacks. "I think they should work and be kept as backups- at least PWC- ZMQ-not so much as it was never tested like PWC- PWC actually worked- if we kept it underwhelmed and on two rigs not three". "i DO NOT want to in any way jeopardize range miner!!!" | NEW owner direction. Making a PWC result count again **CHANGES A RULING** (Beta 2026-07-31) and is the owner's to decide. This brief decides nothing; it maps the work |
| ZMQ is not pursued as a backup | NEW owner inclination. One paragraph on its status is enough |

**Not searched:** VM101's live state, ser8 archives, the private repository, the project PDFs.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** record `git rev-parse HEAD` in your first step.
**Review class:** document. Beta reads it after commit. No relay in this session.
**Report:** `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md`, in your worktree.

**The yardstick (CLAUDE.md).** TFM is judged by prediction-pool hit rate against `k/1000` on
unseen draws. A backup engine is judged by whether the later steps can use its output the
same way, and by that hit rate. Not by matching another engine's numbers.

## 0. Rules for this session

- **Read-only. You edit no existing file.** The only file you write is the report.
- **Worktree:** `git worktree add --detach /home/michael/tfm_worktrees/pwc-backup HEAD`
  (hyphen, not underscore). Write only the report, only there.
- **No commit, push, branch or tag. No fleet, no GPU, no SSH to any rig. No pipeline step.**
  You run no engine. You may run pure functions and existing CPU-only tests in the worktree
  if they write nothing outside it.
- **Live databases:** read-only, as `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md`
  §A says, or not at all.
- **Another session may be editing** in `/home/michael/tfm_worktrees/miner-start`. Leave it alone.
- **Before removing or downgrading anything in your conclusions, cite the document that
  explains why it exists.** This code is over a year old and its reasons are written down.
- **Every claim carries `file:line` read in this session.** Say what you ran and what you
  only read. A keyword hit is not a finding until you have read the text around it.
- Keep going on small things. Stop only if a question cannot be answered without editing
  or running an engine; then say so and move to the next question.

## 1. Questions

**P1 — The record.** Every ruling, proposal, chapter, audit, changelog and exchange file that
decides PWC's status. Find the 2026-07-31 ruling itself and quote what it retired PWC from,
and for what scope. Find the quarantine ruling and its stated condition for lifting. Say
whether any document records RANGE-MINER's design intent that PWC and ZMQ stay as
fallbacks, and where.

**P2 — What PWC does today.** By reading, the path of one Step 1 trial on the PWC route:
what runs, on which hosts, with what concurrency, what it returns, where it can fail, and
what it writes. Constant-skip and hybrid separately.

**P3 — The gaps between today and a backup whose result can count.** For each: what is
missing, the evidence, the files a fix would touch, and whether any of those files is
shared with RANGE-MINER or is in the rig parity set (`scripts/gate12_parity_gate.py:174`,
`:189`). At least:

1. Completion accounting carried from the pass to the trial (Beta's Q1).
2. The seed start.
3. The hybrid quarantine: its cause, and what would have to be shown to lift it.
4. Publication: does PWC's output reach the same certified 22-array generation through the
   same door, so that Steps 2 to 6 cannot tell which engine produced it?
5. What the last year added to RANGE-MINER that PWC lacks: the window anchor and generator
   phase (Brief I), per-direction thresholds, the session rule, dataset authority, the skip
   range. For each, does PWC honour it?
6. Anything else you find.

**P4 — The operating history.** The owner states PWC worked when kept under full load on two
rigs. Find the record: which runs completed, on how many rigs and GPUs, at what settings;
which runs reset or faulted, and at what load. Put it in one table with sources. From that
record only, state the largest load PWC is shown to have carried cleanly. Invent no limit.
If the record does not support a number, say so.

**P5 — Risk to RANGE-MINER.** For each fix in P3: can it be made without touching a file
RANGE-MINER loads? Where it cannot, say exactly what is shared and how the change could be
kept from altering RANGE-MINER's behaviour. Name the fixes that are safe before Run 4 and
those that must wait until after it.

**P6 — Options, not implemented.** Two or three ways to make PWC a real backup, from the
smallest (constant-skip only, two rigs, counts as a backup result) to the fullest. For each:
what it delivers, what it costs, which ruling it changes, what the owner must decide, and
what evidence would show it works. Recommend nothing that P1 to P5 do not support.

**P7 — ZMQ.** One paragraph: what it is, whether any run on record used it, its status.

## 2. Report

`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md`, opening with `## Prior record` (your full
search from P1). Then a ten-line summary in plain words for the owner, P1 to P7, what was
`UNAVAILABLE`, and the commands you used. End with the report's sha256.
