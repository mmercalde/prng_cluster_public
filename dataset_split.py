#!/usr/bin/env python3
"""
S119 Patch 2: Split daily3.json into mid-day and evening subsets.

Produces:
  daily3_midday.json   — mid-day draws only
  daily3_evening.json  — evening draws only

Both output files are identical in format to daily3.json and are
compatible with all pipeline loaders (sieve_filter, window_optimizer,
full_scoring_worker).

daily3.json is NOT modified.

Deploy:
  scp ~/Downloads/dataset_split.py rzeus:~/distributed_prng_analysis/
  ssh rzeus "cd ~/distributed_prng_analysis && \
    source ~/venvs/torch/bin/activate && \
    python3 dataset_split.py"

Pipeline usage after split:
  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
    --start-step 1 --end-step 3 \
    --params '{"lottery_file": "daily3_midday.json"}'

  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
    --start-step 1 --end-step 3 \
    --params '{"lottery_file": "daily3_evening.json"}'
"""

import json
import sys
from pathlib import Path
from collections import Counter

SOURCE = Path("daily3.json")

if not SOURCE.exists():
    print(f"ERROR: {SOURCE} not found. Run from ~/distributed_prng_analysis/", file=sys.stderr)
    sys.exit(1)

raw = json.load(open(SOURCE))
print(f"Loaded {len(raw)} draws from {SOURCE}")

if not raw:
    print("ERROR: empty file", file=sys.stderr)
    sys.exit(1)

# ── Detect format and split ───────────────────────────────────────────────────
if isinstance(raw[0], dict) and 'session' in raw[0]:
    # Confirmed format: list of dicts with session field
    midday  = [r for r in raw if r.get('session') == 'midday']
    evening = [r for r in raw if r.get('session') == 'evening']
    unknown = [r for r in raw if r.get('session') not in ('midday', 'evening')]

    if unknown:
        print(f"WARNING: {len(unknown)} draws have unknown session value — excluded from both files")
        sessions = Counter(r.get('session') for r in unknown)
        print(f"  Unknown session values: {dict(sessions)}")

    method = "session field"

else:
    # Fallback: positional split (even index = midday, odd index = evening)
    # Based on scraper ordering: midday drawn before evening each day
    print("WARNING: No 'session' field found — falling back to positional split")
    print("  Even indices (0,2,4,...) → midday")
    print("  Odd  indices (1,3,5,...) → evening")
    print("  Verify this matches your data ordering before using split files.")
    midday  = raw[0::2]
    evening = raw[1::2]
    method = "positional fallback (even=midday, odd=evening)"

# ── Write output files ────────────────────────────────────────────────────────
out_midday  = Path("daily3_midday.json")
out_evening = Path("daily3_evening.json")

json.dump(midday,  open(out_midday,  'w'), indent=2)
json.dump(evening, open(out_evening, 'w'), indent=2)

# ── Report ────────────────────────────────────────────────────────────────────
print()
print(f"Split method : {method}")
print(f"Midday draws : {len(midday):>6}  →  {out_midday}")
print(f"Evening draws: {len(evening):>6}  →  {out_evening}")
print(f"Total accounted for: {len(midday) + len(evening)} of {len(raw)}")
print()

# Sanity checks
if len(midday) == 0:
    print("WARNING: midday file is empty — check session field values in source data")
if len(evening) == 0:
    print("WARNING: evening file is empty — check session field values in source data")

# Verify draw range in each file
for name, subset in [("midday", midday), ("evening", evening)]:
    if not subset:
        continue
    if isinstance(subset[0], dict):
        draws = [r['draw'] for r in subset]
    else:
        draws = [int(r) for r in subset]
    print(f"{name}: draw range {min(draws)}-{max(draws)}, "
          f"unique values {len(set(draws))}, "
          f"date range {subset[0].get('date','?')} to {subset[-1].get('date','?')}")

print()
print("Both files are compatible with all pipeline loaders.")
print("daily3.json was NOT modified.")
