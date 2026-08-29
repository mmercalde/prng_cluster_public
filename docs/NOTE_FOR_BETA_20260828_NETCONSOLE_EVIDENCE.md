# NOTE FOR TEAM BETA — netconsole evidence correction (INFORMATIONAL)

**Dated 2026-08-28 · Team Alpha · NO RULING REQUESTED · Does not affect the R2 approval.**

**Instruction to whoever assembles the next Beta package:** fold the paragraph below into that
submission as an informational note. Do not send this file on its own, and do not add it to a
question list — nothing here needs a decision.

---

## The paragraph

During the Run-4 post-commit runway (R2 ruling step 3, netconsole status), we found that
`docs/RIG_CRASH_FORENSIC_20260822.md` overstates one unavailable-observer line. It records
`netconsole = EMPTY`; the capture actually holds **11 packets dated 2026-08-22** from all three
Proxmox hosts. The operator has confirmed all 11 are **post-incident cleanup** — he always shuts
the rigs down after a crash and did so that evening — so the `NC-TEST3`/`NCPROOF` pair is his
arm-verification test and the `systemd-shutdown` lines are that shutdown; the
`watchdog did not stop!` line on `.155` belongs to an orderly shutdown path, not a fault. The
correction is therefore narrow but worth having on the record: the doc's disjunction was
*"no event"* vs *"not active"*, and the `NCPROOF` packets **close the "not active" branch** —
netconsole was armed and delivering on all three hosts — leaving *"no event during the freeze
itself"* as the correct and only reading. **The 2026-08-22 mid-run freeze remains UNDETERMINED**;
an armed-but-silent netconsole shows no kernel message reached the wire, which is not evidence of
a healthy host, and the document's audit-claim scope ("NOT a claim about why the hosts stopped")
is unaffected. Kernel monotonic uptime stamps additionally bound the freeze to **18:42-18:52** —
implied boot times of 18:52:41/18:58:18/18:58:55 all post-date the run log's last write at
18:42:04, and uptime runs continuous from the 19:25 arming to each shutdown (drift <0.07 s), so
those packets come from post-incident boot sessions rather than from the frozen one. Recorded as
`docs/LEADS.md` L-3, with a **CORRECTION ADDENDUM appended** to the forensic document (original
text unedited) and a re-arm runbook stub at `docs/RUNBOOK_NETCONSOLE_REARM.md`. Nothing here
changes the Route-A patch, the R2 approval, or the Run-4 claim boundary; we raise it only because
a committed forensic document is now known to be inaccurate on one line and Beta reads that
document.

---

## Context for the assembler (not for Beta)

- **Why informational, not a submission.** The finding corrects a fact in an Alpha-authored
  forensic document. It touches no ruling, no gate, no production code, and no Run-4 decision.
- **Why it is raised at all rather than silently fixed.** Beta reads
  `RIG_CRASH_FORENSIC_20260822.md`, and a known-inaccurate line in a committed governance document
  should not be corrected only in a place Beta will not look.
- **What must not be claimed from it.** That the freeze is explained, that any component is
  exonerated, or that netconsole will capture the next freeze. None of those follows.
- **Related open item, already on Beta's recorded-but-unrepaired list:** the rig kernel-log access
  gap (unprivileged LXC, no root key auth to the Proxmox hosts) is what forced this evidence to
  come from an archived capture rather than a live probe.
