# TEAM ALPHA → TEAM BETA — SAFETY NOTICE: the scraper's `--recent` path destroys the dataset

**Not a ruling request. An operational hazard Beta should hold independently of 6-P2's approval
timeline.**

**Source:** `ser8:~/Downloads/daily3_scraper.py`, **Revision 1.5**, read this session.
`ser8:~/cluster_controller/daily3_scraper.py` is **byte-identical**. This is the revision
`pa_pick3_scraper.py`'s header names as its parent, so the lineage is confirmed.

**Note the audit surface:** `daily3_scraper.py` has **never been in git history** and is **not
gitignored** — the program that produces the canonical dataset has never been under version
control. It is now being tracked. No repo-scoped audit could previously have seen any of what
follows.

---

## 1. The defect

`main()` accumulates scraped rows into `all_draws`, then:

```python
Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))   # OUTPUT_FILE = "daily3.json"
```

**It never reads the existing `daily3.json`.** It writes only what it just scraped, over the top.

And:

```python
if recent:
    start_year = end_year = TODAY.year
```

**Therefore `python3 daily3_scraper.py --json --recent` replaces the entire canonical dataset with
the current year alone** — roughly 500 records in place of 18,068. No merge, no dedup, no
append. No error. The exit status is success.

There is also **no atomic boundary**: `write_text` is not temp-then-`os.replace`, so a crash or a
full disk mid-write truncates `daily3.json` outright.

## 2. Why this has not already happened

`daily3scraper.service` has been **enabled since Sep 2025 with `Restart=always`**, targeting
`run_daily3scraper.py` — **a file that has never existed** — and has looped ENOENT on every boot
since.

**That loop is the only thing that has prevented this.** `--recent` is precisely the invocation a
daily service would use.

**The item we had recorded as a defect was, in effect, a safety interlock.** Had anyone tidied the
unit by repointing `ExecStart` at the real scraper — an obvious, well-intentioned repair — the
canonical dataset would have been truncated on the next boot, silently, and the loss would have
surfaced later as an unexplained change in every downstream result.

## 3. What Alpha asks Beta to hold

1. **`daily3scraper.service` must not be repaired, repointed or re-enabled until 6-P2 is
   certified** — and then only against a target that has been verified to exist and to be
   append-only. This holds regardless of when 6-P2 is approved.
2. **No process may invoke `daily3_scraper.py` against the real output path** in the interim.
   Scraping into a temp directory is fine; publishing is not.
3. **Removing the destructive path is a primary purpose of 6-P2**, not incidental cleanup. Alpha
   has made it 6-P2's first gate: `G-DESTRUCTIVE-PATH-GONE` publishes a full dataset, then
   publishes a current-year-only scrape, and requires the published version to still contain the
   full history.

**No dataset loss is known to have occurred.** The current `daily3.json` record count should be
confirmed against the frozen manifest as routine assurance; Alpha has not done so from the
sandbox, and states that plainly rather than implying it has been checked.

## 4. VIR declaration

- **audit claim scope:** the scraper characterization is from the **ser8 Revision 1.5 file**, not
  from a repository file — **none exists**.
- **searched surfaces:** tracked repo (absent, confirmed via `git log --all --diff-filter=A` and
  `git check-ignore`); ser8 `~/Downloads/` and `~/cluster_controller/` via Michael.
- **unavailable surfaces:** whether a **divergent** copy exists on VM101 or Zeus — 6-P2 §1 closes
  this, and a divergent copy would mean a second lineage and a different target. The claim
  "Revision 1.5 is the scraper" is therefore **[UNVERIFIED] against host state** at the time of
  writing.
