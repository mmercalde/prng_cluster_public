# Revision 1.1: PA Pick 3 scraper — adapted from CA daily3_scraper.py Rev 1.5
# Changes from CA version:
#   - BASE_URL: /ca/daily-3  →  /pa/pick-3
#   - OUTPUT_FILE: daily3.json  →  pa_pick3.json
#   - Start year: 2000  →  2002  (PA Pick 3 midday began ~2002; evening earlier)
#   - Description strings updated for PA
#
# Rev 1.1 fix — PA Wild Ball:
#   LotteryCorner encodes PA draws as NNNWWild (e.g. "9083Wild", "4767Wild")
#   where NNN = the 3-digit Pick 3 result and W = the Wild Ball digit (irrelevant to pipeline).
#   The 4-digit prefix is draw(3) + wildball(1) concatenated with no separator.
#   Fix: strip "Wild" suffix (case-insensitive), then take first 3 characters as the draw.
#   Wild ball digit is discarded — not part of the PRNG output sequence we model.
#
# OUTPUT FORMAT: identical to daily3.json — fully pipeline-compatible:
#   [{"date": "YYYY-MM-DD", "session": "midday"|"evening", "draw": int}, ...]
#   sorted chronological, deduped on (date, session, draw)
#
# VERIFY URL on Zeus before first full run:
#   curl -s "https://www.lotterycorner.com/pa/pick-3-midday/2025" | grep -i "<table" | head -3
#   If 404/redirect, try: /pa/daily-3-midday/2025
#
# Deploy to Zeus:
#   scp ~/Downloads/pa_pick3_scraper.py rzeus:~/distributed_prng_analysis/
#   ssh rzeus "cd ~/distributed_prng_analysis && \
#       source ~/venvs/torch/bin/activate && \
#       python3 pa_pick3_scraper.py --json"
#
# Pipeline usage:
#   PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
#       --start-step 1 --end-step 1 \
#       --params '{"lottery_file": "pa_pick3.json"}'
#
# Session split (reuse existing dataset_split.py with SOURCE override):
#   python3 dataset_split.py --source pa_pick3.json
#   (or run directly — edit SOURCE at top of dataset_split.py)

import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import datetime as dt
from dateutil import parser as date_parser
import argparse

# --- Configuration ---
BASE_URL = "https://www.lotterycorner.com/pa/pick-3"   # PA uses "pick-3" not "daily-3"
DRAW_TYPES = ["midday", "evening"]
TODAY = dt.date.today()
OUTPUT_FILE = "pa_pick3.json"

# PA Pick 3 history notes:
#   - Evening draws available from ~2000
#   - Midday draws added ~2002
#   - Default start 2000 — midday years pre-2002 will return 0 draws (handled gracefully)

# --- Functions ---
def parse_draw_value(raw: str) -> int:
    """
    Parse PA Pick 3 draw value from LotteryCorner raw cell text.

    PA-specific: LotteryCorner appends the Wild Ball digit directly to the
    3-digit draw result with no separator, then adds 'Wild' suffix.
    Examples:
        '9083Wild' → draw=908, wildball=3  (take first 3 chars)
        '4767Wild' → draw=476, wildball=7
        '0402Wild' → draw=040 = 40
        '123'      → draw=123  (no wild ball, e.g. older records)
        '0'        → draw=0    (edge case)

    Always strips 'Wild' suffix (case-insensitive) then takes first 3 digits.
    Raises ValueError if result is not a valid 0-999 integer.
    """
    s = raw.strip().replace(" ", "").replace("\n", "")
    # Strip Wild suffix (case-insensitive)
    if s.lower().endswith("wild"):
        s = s[:-4]  # remove 'Wild'
    # Take first 3 characters as the draw (discard wild ball digit if present)
    s = s[:3]
    draw = int(s)
    if not (0 <= draw <= 999):
        raise ValueError(f"Draw value {draw} out of range 0-999")
    return draw


def fetch_draws(year, draw_type):
    """
    Fetch draws for a given year and draw type.
    Rev 1.1: Uses parse_draw_value() to handle PA Wild Ball format.
    """
    url = f"{BASE_URL}-{draw_type}/{year}"
    print(f"🔍 Fetching: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"❌ No table found in {url}")
        return []

    table_body = table.find("tbody")
    if not table_body:
        print(f"❌ No tbody found in {url}")
        return []

    rows = table_body.find_all("tr")
    print(f"🔍 Found {len(rows)} table rows in tbody.")
    draws = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 2 and cols[0].text.strip() not in ["Date", ""] and cols[1].text.strip() not in ["Result", ""]:
            date_str = cols[0].text.strip()
            draw_str = cols[1].text.strip().replace(" ", "").replace("\n", "")

            if not date_str or not draw_str:
                print(f"Skipping empty data: {row.text.strip()}")
                continue

            try:
                draw = parse_draw_value(draw_str)
                date_obj = date_parser.parse(date_str, fuzzy=True).date()
                draws.append({
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "session": draw_type,
                    "draw": draw
                })
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid row: {date_str} - {draw_str} ({e})")
                continue
        else:
            print(f"Skipping non-data row: {row.text.strip()}")

    if not draws:
        print(f"⚠️  No valid draws parsed for {draw_type} {year}")
    return draws


def deduplicate(draws):
    """Remove exact (date, session, draw) duplicates, preserve chronological order."""
    seen = set()
    deduped = []
    for d in draws:
        key = (d["date"], d["session"], d["draw"])
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped


def main(start_year, end_year, json_output, recent):
    """
    Scrape PA Pick 3 draws from start_year to end_year.
    """
    if recent:
        start_year = end_year = TODAY.year

    print(f"📆 Today is: {TODAY.strftime('%Y-%m-%d')}")
    print(f"Fetching PA Pick 3 draws from {start_year} to {end_year}.")

    all_draws = []
    for year in range(start_year, end_year + 1):
        for draw_type in DRAW_TYPES:
            draws = fetch_draws(year, draw_type)
            all_draws.extend(draws)

    # Sort chronological (date asc, midday before evening within same date)
    session_order = {"midday": 0, "evening": 1}
    all_draws.sort(key=lambda x: (x["date"], session_order.get(x["session"], 2)))

    # Deduplicate
    before_dedup = len(all_draws)
    all_draws = deduplicate(all_draws)
    dupes_removed = before_dedup - len(all_draws)

    print(f"✅ Total draws parsed  : {before_dedup}")
    print(f"   Duplicates removed  : {dupes_removed}")
    print(f"   Final draw count    : {len(all_draws)}")

    if not all_draws:
        print("❌ No draws parsed — check URL or parsing logic.")
        print("   Try verifying on Zeus:")
        print("   curl -s 'https://www.lotterycorner.com/pa/pick-3-midday/2025' | grep -i '<table' | head -3")
        return

    # Draw range sanity check
    draw_values = [d["draw"] for d in all_draws]
    print(f"   Draw range          : {min(draw_values)}–{max(draw_values)}")
    print(f"   Unique draw values  : {len(set(draw_values))}")
    sessions = {d["session"] for d in all_draws}
    for s in sorted(sessions):
        count = sum(1 for d in all_draws if d["session"] == s)
        print(f"   {s:10s}         : {count}")
    if all_draws:
        print(f"   Date range          : {all_draws[0]['date']} to {all_draws[-1]['date']}")

    if json_output:
        try:
            Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))
            print(f"🧊 Stored {len(all_draws)} draws into {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Failed to write {OUTPUT_FILE}: {e}")
    else:
        print("⚠️  JSON output not requested — pass --json to save file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PA Pick 3 draws and store in JSON.")
    parser.add_argument("--start", type=int, default=2000, help="Start year (default: 2000)")
    parser.add_argument("--end",   type=int, default=dt.datetime.now().year, help="End year (default: current year)")
    parser.add_argument("--json",   action="store_true", help="Export to JSON file")
    parser.add_argument("--recent", action="store_true", help="Fetch only current year's draws")
    args = parser.parse_args()
    main(args.start, args.end, args.json, args.recent)
