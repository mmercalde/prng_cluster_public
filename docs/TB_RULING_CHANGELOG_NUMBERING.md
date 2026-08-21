# TB RULING — SESSION-CHANGELOG NUMBERING

**Received:** 2026-08-18
**Responds to:** `docs/TB_RULING_REQUEST_CHANGELOG_NUMBERING.md`
**Recorded by:** Team Alpha, verbatim below.

**Binding convention (until backfill reconciliation):** all new changelogs are
`SESSION_CHANGELOG_YYYYMMDD_<TOPIC>.md`. No new S-numbers. No retro-numbering of the three
existing topic-named sessions. At SER8-backlog import, one deliberate reconciliation pass
with a single explicit Beta ruling: restore the S-sequence (retro-numbering where
determinable) or formally retire it. Date + topic is canonical until then, not a temporary
exception.

---

## Ruling body (verbatim)

TEAM BETA RULING — SESSION-CHANGELOG NUMBERING

Option A is APPROVED and governs immediately.

Use date + topic naming starting with the owed governance changelog:
`SESSION_CHANGELOG_20260817_GOVERNANCE_RULINGS.md`

The reason is straightforward: the repository does not presently contain enough information
to assign a trustworthy next S-number. `S185` is only the highest numbered changelog visible
in the committed record, while approximately 20 SER8-only changelogs still await backfill.
Assigning `S186`, `S187`, etc. now would convert an unknown historical sequence into guessed
governance state and create avoidable collision/renumbering risk.

This is also consistent with the three most recent committed sessions, which already use
date + topic names without claiming session numbers.

Binding convention until backfill reconciliation

For all newly written changelogs, use:
`SESSION_CHANGELOG_YYYYMMDD_<TOPIC>.md`

Do not assign new S-numbers and do not retro-number the three existing topic-named sessions
yet.

When the SER8 backlog is imported, perform one deliberate reconciliation pass using the
complete record. At that point Beta should make a single explicit ruling between:

* reconstructing and restoring the S-number sequence, including retro-numbering the
  topic-named sessions where historically determinable; or
* formally retiring S-numbering and retaining date + topic permanently.

Until that reconciliation occurs, date + topic is the canonical naming scheme, not merely a
temporary exception.

Alpha may write and commit the 2026-08-17 governance changelog under that convention
immediately. No enumeration of the SER8 backlog is required first.
