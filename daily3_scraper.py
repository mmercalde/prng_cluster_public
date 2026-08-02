# Revision 1.5: Modified to handle header/footer rows more robustly and avoid skipping valid data.
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import datetime as dt
from dateutil import parser as date_parser
import argparse

# --- Configuration ---
BASE_URL = "https://www.lotterycorner.com/ca/daily-3"
DRAW_TYPES = ["midday", "evening"]
TODAY = dt.date.today()
OUTPUT_FILE = "daily3.json"

# --- Functions ---
def fetch_draws(year, draw_type):
    """
    Fetch draws for a given year and draw type.
    Revision 1.5: Find the correct table body and iterate through its rows,
    while also handling the 'Date', 'Result', 'Jackpot' header rows and empty rows.
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
    
    # Revision 1.5: Target the table body for more reliable data parsing
    table_body = table.find("tbody")
    if not table_body:
        print(f"❌ No tbody found in {url}")
        return []
    
    rows = table_body.find_all("tr")
    print(f"🔍 Found {len(rows)} table rows in tbody.")
    draws = []
    
    for row in rows:
        cols = row.find_all("td")
        # Ensure we have at least two columns and the first column is not 'Date' or empty
        if len(cols) >= 2 and cols[0].text.strip() not in ["Date", ""] and cols[1].text.strip() not in ["Result", ""]:
            date_str = cols[0].text.strip()
            draw_str = cols[1].text.strip().replace(" ", "").replace("\n", "")
            
            if not date_str or not draw_str:
                print(f"Skipping empty data: {row.text.strip()}")
                continue
            
            try:
                draw = int(draw_str)
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
        print(f"⚠️ No valid draws parsed for {draw_type} {year}")
    return draws

def main(start_year, end_year, json_output, recent):
    """
    Scrape draws from start_year to end_year.
    Revision 1.5: Same as before, but with updated fetch_draws.
    """
    if recent:
        start_year = end_year = TODAY.year
    print(f"📆 Today is: {TODAY.strftime('%Y-%m-%d')}")
    print(f"Fetching draws from {start_year} to {end_year}.")
    
    all_draws = []
    for year in range(start_year, end_year + 1):
        for draw_type in DRAW_TYPES:
            draws = fetch_draws(year, draw_type)
            all_draws.extend(draws)
    
    print(f"✅ Total draws parsed: {len(all_draws)}")
    if not all_draws:
        print("❌ No draws parsed, check website or parsing logic.")
        return
    if json_output:
        try:
            Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))
            print(f"🧊 Stored {len(all_draws)} draws into {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Failed to write {OUTPUT_FILE}: {e}")
    else:
        print("⚠️ JSON output not requested, no file saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape CA Daily 3 draws and store in JSON.")
    parser.add_argument("--start", type=int, default=2000, help="Start year (e.g., 2000)")
    parser.add_argument("--end", type=int, default=dt.datetime.now().year, help="End year (e.g., 2025)")
    parser.add_argument("--json", action="store_true", help="Export JSON")
    parser.add_argument("--recent", action="store_true", help="Fetch only current year's draws")
    args = parser.parse_args()
    main(args.start, args.end, args.json, args.recent)
