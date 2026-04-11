"""
Fetch all games for a chess.com username and save them to data/raw/<username>_games.json.
Usage: python scripts/download_data.py [belindafails]
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_url(url):
    """Fetch URL and return the body as a string"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ChessDataDownloader/1.0 (Python)"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def get_archive_list(username):
    """Returns a list of monthly archive URLs"""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    print(f"Fetching archive list from: {url}")
    text = fetch_url(url)
    data = json.loads(text)
    archives = data.get("archives", [])
    return archives


def get_games_from_archive(archive_url):
    """Fetch one monthly archive and return its list of games"""
    text = fetch_url(archive_url)
    data = json.loads(text)
    games = data.get("games", [])
    return games


def download_games(username, output_file=None):
    """Download all games for chess.com user and write them to a single JSON file"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if output_file is None:
        output_file = RAW_DIR / f"{username}_games.json"
    else:
        output_file = Path(output_file)

    print(f"Downloading games for user: {username}")
    print("Longer if the player has many months of games.\n")

    try:
        archive_list = get_archive_list(username)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Error: Username '{username}' not found on Chess.com.")
        else:
            print(f"Error: HTTP {e.code} when fetching archives.")
        return
    except urllib.error.URLError as e:
        print(f"Error: Could not reach Chess.com. Check your internet connection. {e}")
        return

    if not archive_list:
        print("This player has no game archives (no games played).")
        return

    first_parts = archive_list[0].rstrip("/").split("/")
    last_parts  = archive_list[-1].rstrip("/").split("/")
    first_month = f"{first_parts[-2]}-{first_parts[-1]}" if len(first_parts) >= 2 else archive_list[0]
    last_month  = f"{last_parts[-2]}-{last_parts[-1]}"  if len(last_parts)  >= 2 else archive_list[-1]
    print(f"  {username}  |  {len(archive_list)} months  |  {first_month} → {last_month}\n")

    all_games = []
    for i, archive_url in enumerate(archive_list):
        print(f"  Fetching archive {i + 1} of {len(archive_list)}: {archive_url}")
        try:
            games = get_games_from_archive(archive_url)
            all_games.extend(games)
            print(f"    {len(games)} games")
        except urllib.error.HTTPError as e:
            print(f"    Skipped — HTTP {e.code}" + (" (rate limited)" if e.code == 429 else ""))
        except urllib.error.URLError as e:
            print(f"    Skipped — network error: {e}")

    result = {
        "username": username,
        "total_games": len(all_games),
        "games": all_games
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_games)} games for '{username}' to: {output_file}")


def main():
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else "belindafails"
    download_games(username)


if __name__ == "__main__":
    main()
