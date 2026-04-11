"""
Convert raw chess.com JSON (from download_data.py) into a flat CSV for the dashboard.
Usage: python scripts/process_data.py [belindafails]
"""

import json
import csv
import re
from pathlib import Path
from datetime import datetime

# paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# columns to write to CSV 
EXTRACT_COLUMNS = [
    "game_url",
    "end_time",
    "end_date",
    "white_username",
    "white_rating",
    "white_result",
    "black_username",
    "black_rating",
    "black_result",
    "time_class",
    "time_control",
    "rules",
    "rated",
    "white_accuracy",
    "black_accuracy",
    # derived columns
    "my_username",
    "my_color",
    "my_rating",
    "opponent_username",
    "opponent_rating",
    "my_result",
    "opponent_result",
    "rating_diff",
    # opening-related
    "eco",
    "eco_code",
    "opening_name",
]


def safe_get(obj, *keys, default=""):
    """Safely traverse a nested dict — e.g. safe_get(game, "white", "rating")"""
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return default
    return obj if obj is not None else default


def try_extract_eco_code(eco_url):
    """Pull an ECO code (e.g. B12) from a URL if one is present"""
    if not eco_url or not isinstance(eco_url, str):
        return ""
    # Some APIs put ECO in path or query; match patterns like A00, B12, C30
    match = re.search(r"\b([A-E]\d{2})\b", eco_url, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def game_to_row(game, username):
    """Flatten one game dict into a CSV row"""
    end_time = safe_get(game, "end_time", default=None)
    if end_time:
        try:
            end_date = datetime.utcfromtimestamp(int(end_time)).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            end_date = ""
    else:
        end_date = ""

    pgn = safe_get(game, "pgn", default="")
    opening_name = ""
    eco_code = ""
    if pgn:
        m = re.search(r'\[Opening "(.+?)"\]', pgn)
        if m:
            opening_name = m.group(1)
        m = re.search(r'\[ECO "([A-E]\d{2})"\]', pgn)
        if m:
            eco_code = m.group(1)
    eco_raw = safe_get(game, "eco", default="")
    eco = eco_raw if isinstance(eco_raw, str) else str(eco_raw)
    if not eco_code:
        eco_code = try_extract_eco_code(eco)

    white_user = safe_get(game, "white", "username")
    black_user = safe_get(game, "black", "username")
    white_rating = safe_get(game, "white", "rating")
    black_rating = safe_get(game, "black", "rating")
    white_result = safe_get(game, "white", "result")
    black_result = safe_get(game, "black", "result")

    if white_user == username:
        my_color = "white"
        my_rating = white_rating
        opponent_username = black_user
        opponent_rating = black_rating
        my_result = white_result
        opponent_result = black_result
    elif black_user == username:
        my_color = "black"
        my_rating = black_rating
        opponent_username = white_user
        opponent_rating = white_rating
        my_result = black_result
        opponent_result = white_result
    else:
        my_color = ""
        my_rating = ""
        opponent_username = ""
        opponent_rating = ""
        my_result = ""
        opponent_result = ""

    rating_diff = ""
    try:
        mr = int(my_rating) if my_rating != "" else None
        opp = int(opponent_rating) if opponent_rating != "" else None
        if mr is not None and opp is not None:
            rating_diff = mr - opp
    except (ValueError, TypeError):
        pass

    row = {
        "game_url": safe_get(game, "url"),
        "end_time": end_time,
        "end_date": end_date,
        "white_username": white_user,
        "white_rating": white_rating,
        "white_result": white_result,
        "black_username": black_user,
        "black_rating": black_rating,
        "black_result": black_result,
        "time_class": safe_get(game, "time_class"),
        "time_control": safe_get(game, "time_control"),
        "rules": safe_get(game, "rules"),
        "rated": safe_get(game, "rated"),
        "white_accuracy": safe_get(game, "accuracies", "white"),
        "black_accuracy": safe_get(game, "accuracies", "black"),
        "my_username": username,
        "my_color": my_color,
        "my_rating": my_rating,
        "opponent_username": opponent_username,
        "opponent_rating": opponent_rating,
        "my_result": my_result,
        "opponent_result": opponent_result,
        "rating_diff": rating_diff,
        "eco": eco,
        "eco_code": eco_code,
        "opening_name": opening_name,
    }
    return row


def process_raw_json(input_path=None, output_path=None, username=None):
    """Convert a raw JSON file into a processed CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if input_path is None:
        raw_files = list(RAW_DIR.glob("*_games.json"))
        if not raw_files:
            print("No raw JSON found in data/raw/. Run scripts/download_data.py first.")
            return
        input_path = max(raw_files, key=lambda p: p.stat().st_mtime)
        print(f"Using raw file: {input_path}")

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    games = data.get("games", [])
    if username is None:
        username = data.get("username", "unknown")

    print(f"Loaded {len(games)} games for username: {username}")

    rows = []
    for game in games:
        row = game_to_row(game, username)
        rows.append({k: row.get(k, "") for k in EXTRACT_COLUMNS})

    if output_path is None:
        output_path = PROCESSED_DIR / f"{username}_games.csv"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXTRACT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output_path}")


def main():
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "belindafails"
    input_path = RAW_DIR / f"{username}_games.json"
    output_path = PROCESSED_DIR / f"{username}_games.csv"
    process_raw_json(input_path=input_path, output_path=output_path, username=username)


if __name__ == "__main__":
    main()
