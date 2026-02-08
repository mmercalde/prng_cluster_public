# SESSION CHANGELOG — [DATE] (S[NUMBER])

**Focus:** [One-line description of session goal]  
**Outcome:** [One-line result — what was accomplished or what happened]

---

## Summary

[2-3 sentences max. What was done, what was the result, any surprises.]

---

## Work Completed

| Item | Status | Notes |
|------|--------|-------|
| [Task 1] | ✅ | [Brief note] |
| [Task 2] | ✅ | [Brief note] |
| [Task 3] | ⚠️ Partial | [What remains] |

---

## Files Created/Modified

| File | Action | Destination |
|------|--------|-------------|
| `filename.py` | Created | `zeus:~/distributed_prng_analysis/` |
| `filename.md` | Created | `zeus:~/distributed_prng_analysis/docs/` |
| `existing_file.py` | Modified | Lines X-Y changed |

---

## Issues / Incidents

[Only if applicable. Delete section if clean session.]

| Issue | Resolution |
|-------|------------|
| [What happened] | [How it was fixed] |

---

## Decisions Made

[Only if applicable. Delete section if no decisions.]

- [Decision and rationale]

---

## Copy Commands

```bash
# From ser8 Downloads to Zeus
scp ~/Downloads/[file1] rzeus:~/distributed_prng_analysis/[dest]/
scp ~/Downloads/[file2] rzeus:~/distributed_prng_analysis/docs/
scp ~/Downloads/SESSION_CHANGELOG_[DATE]_S[NUMBER].md rzeus:~/distributed_prng_analysis/docs/

# Verification on Zeus (if applicable)
ssh rzeus
cd ~/distributed_prng_analysis
[verification commands]
```

---

## Git Commands

```bash
cd ~/distributed_prng_analysis
git add [files]
git commit -m "[type]: [description]

[body if needed]"
git push origin main
```

---

## Hot State (Next Session Pickup)

**Where we left off:** [Exactly what was last done / what's running]  
**Next action:** [The literal next thing to do]  
**Blockers:** [None / or what's blocking]  
**Context needed:** [Any file or output the next session should look at first]

---

*End of Session [NUMBER]*
