"""
fetch_boxscores.py

Phase 1 data script: pulls NBA player box scores via nba_api and keeps a
persistent CSV up to date.

Designed to be re-run daily:
  - First run (no existing file): backfills full historical seasons.
  - Every run after: re-fetches ONLY the current season (one API call) and
    replaces just that season's rows in the store. Finalized prior seasons
    are never re-fetched.

Re-fetching the whole current season each run -- rather than trying to pull
a precise "since yesterday" delta -- is a deliberate, pragmatic choice: it's
still cheap (a few thousand rows, one API call), it's idempotent (safe to
run more than once a day), and it self-corrects for the NBA's occasional
post-game stat corrections, which a true append-only delta wouldn't catch.

Usage:
    python fetch_boxscores.py                # incremental update (or backfill if first run)
    python fetch_boxscores.py --full-refresh  # force a full historical re-backfill
"""

import argparse
import os
import time
from datetime import date

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playerindex


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "nba_live_data")
SCORES_DIRECTORY = os.path.join(DATA_DIRECTORY, "scores")
OUTPUT_PATH = os.path.join(SCORES_DIRECTORY, "player_boxscores.csv")

SEASONS_BACK = 1           # how many completed seasons to backfill on first run
REQUEST_DELAY_SECONDS = 1  # be polite to the API during multi-season backfills


def get_current_season_start_year(today: date) -> int:
    """NBA seasons run Oct -> Jun. Aug onward belongs to the season
    starting that calendar year; Jan-Jul belongs to the one before it."""
    return today.year if today.month >= 8 else today.year - 1


def season_str(start_year: int) -> str:
    """e.g. 2025 -> '2025-26', matching nba_api's expected season format."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_season(season: str) -> pd.DataFrame:
    print(f"  Fetching {season}...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"    -> {len(df)} player-game rows")
    return df


def full_backfill(current_start_year: int) -> pd.DataFrame:
    seasons = [season_str(y) for y in range(current_start_year - SEASONS_BACK, current_start_year + 1)]
    frames = []
    for season in seasons:
        try:
            frames.append(fetch_season(season))
        except Exception as e:
            print(f"    -> Error fetching {season}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)


    frames = [df.dropna(axis=1, how='all') for df in frames if not df.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="last")
    return combined.sort_values(["GAME_DATE", "TEAM_ID", "PLAYER_ID"]).reset_index(drop=True)


def incremental_update(existing: pd.DataFrame, current_start_year: int, current_season: str) -> pd.DataFrame:
    # Split off everything that ISN'T the current season -- those seasons
    # are finalized and never need to be touched again.
    season_of_row = np.where(
        existing["GAME_DATE"].dt.month >= 8,
        existing["GAME_DATE"].dt.year,
        existing["GAME_DATE"].dt.year - 1,
    )
    prior_seasons = existing[season_of_row != current_start_year]

    # Re-pull the current season fresh and wholesale-replace it. This
    # naturally picks up new games AND any stat corrections to recent ones.
    fresh_current = fetch_season(current_season)

    combined = pd.concat([prior_seasons, fresh_current], ignore_index=True)
    combined = combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="last")
    return combined.sort_values(["GAME_DATE", "TEAM_ID", "PLAYER_ID"]).reset_index(drop=True)


def update_boxscores(force_full_refresh: bool = False) -> pd.DataFrame:
    os.makedirs(SCORES_DIRECTORY, exist_ok=True)
    today = date.today()
    current_start_year = get_current_season_start_year(today)
    current_season = season_str(current_start_year)

    existing = None
    if os.path.exists(OUTPUT_PATH) and not force_full_refresh:
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=["GAME_DATE"])

    if existing is None:
        print("No existing store found (or --full-refresh set) -- running full historical backfill.")
        combined = full_backfill(current_start_year)
    else:
        print(f"Existing store found ({len(existing)} rows). Refreshing current season ({current_season}) only.")
        combined = incremental_update(existing, current_start_year, current_season)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(combined)} total rows to {OUTPUT_PATH}")
    return combined

def update_player_positions_cache(data_dir: str):
    """
    Fetch player positions using nba_api's PlayerIndex endpoint
    and save them to a local CSV.

    This function:
      - Reads PLAYER_ID / PLAYER_NAME from player_boxscores.csv
      - Makes ONE NBA API request for the player index
      - Extracts PLAYER_ID and POSITION
      - Saves the result to player_positions.csv
      - NEVER modifies player_boxscores.csv

    Existing cached positions are preserved when the NBA API does
    not return a position for a player.
    """

    scores_path = os.path.join(
        data_dir,
        "scores",
        "player_boxscores.csv"
    )

    output_path = os.path.join(
        data_dir,
        "player_positions.csv"
    )

    # ---------------------------------------------------------
    # 1. Make sure the box-score database exists
    # ---------------------------------------------------------

    if not os.path.exists(scores_path):
        print(f"⚠️ Box-score file not found: {scores_path}")
        return

    print("Updating player positions cache via NBA PlayerIndex...")

    # ---------------------------------------------------------
    # 2. Read players from the box-score database
    #
    # IMPORTANT:
    # This only READS player_boxscores.csv.
    # It does not modify it.
    # ---------------------------------------------------------

    boxscores = pd.read_csv(
        scores_path,
        usecols=["PLAYER_ID", "PLAYER_NAME"]
    )

    players = (
        boxscores[
            ["PLAYER_ID", "PLAYER_NAME"]
        ]
        .drop_duplicates(subset=["PLAYER_ID"])
        .copy()
    )

    players["PLAYER_ID"] = pd.to_numeric(
        players["PLAYER_ID"],
        errors="coerce"
    )

    players = players.dropna(subset=["PLAYER_ID"])

    players["PLAYER_ID"] = players["PLAYER_ID"].astype(int)

    print(
        f"Found {len(players)} unique players "
        f"in player_boxscores.csv"
    )

    # ---------------------------------------------------------
    # 3. Load existing position cache
    # ---------------------------------------------------------

    if os.path.exists(output_path):

        existing = pd.read_csv(output_path)

        required_columns = {
            "PLAYER_ID",
            "PLAYER_NAME",
            "POSITION",
        }

        if required_columns.issubset(existing.columns):

            existing["PLAYER_ID"] = pd.to_numeric(
                existing["PLAYER_ID"],
                errors="coerce"
            )

            existing = existing.dropna(
                subset=["PLAYER_ID"]
            )

            existing["PLAYER_ID"] = (
                existing["PLAYER_ID"].astype(int)
            )

            existing = existing.drop_duplicates(
                subset=["PLAYER_ID"],
                keep="last"
            )

        else:

            print(
                "⚠️ Existing player_positions.csv has "
                "unexpected columns. Rebuilding it."
            )

            existing = pd.DataFrame(
                columns=[
                    "PLAYER_ID",
                    "PLAYER_NAME",
                    "POSITION",
                ]
            )

    else:

        existing = pd.DataFrame(
            columns=[
                "PLAYER_ID",
                "PLAYER_NAME",
                "POSITION",
            ]
        )

    # ---------------------------------------------------------
    # 4. Fetch the NBA player index
    #
    # This is ONE API request rather than one request per player.
    # ---------------------------------------------------------

    try:

        print("  Fetching NBA PlayerIndex...")

        response = playerindex.PlayerIndex(
            league_id="00",
            season="2025-26",
            historical_nullable="0",
            timeout=30,
        )

        nba_players = response.get_data_frames()[0]

        print(
            f"  -> NBA returned "
            f"{len(nba_players)} players"
        )

    except Exception as e:

        print(
            f"⚠️ Failed to fetch NBA PlayerIndex: {e}"
        )

        # Don't destroy a good existing cache if the API
        # happens to be unavailable today.
        if os.path.exists(output_path):
            print(
                "Keeping existing player_positions.csv."
            )

        return

    # ---------------------------------------------------------
    # 5. Verify the response contains the fields we need
    # ---------------------------------------------------------

    required_nba_columns = {
        "PERSON_ID",
        "POSITION",
    }

    missing_columns = (
        required_nba_columns - set(nba_players.columns)
    )

    if missing_columns:

        print(
            "⚠️ NBA PlayerIndex response is missing "
            f"columns: {missing_columns}"
        )

        print(
            f"Available columns: "
            f"{list(nba_players.columns)}"
        )

        return

    # ---------------------------------------------------------
    # 6. Extract only the data we need
    # ---------------------------------------------------------

    nba_positions = nba_players[
        ["PERSON_ID", "POSITION"]
    ].copy()

    nba_positions = nba_positions.rename(
        columns={
            "PERSON_ID": "PLAYER_ID"
        }
    )

    nba_positions["PLAYER_ID"] = pd.to_numeric(
        nba_positions["PLAYER_ID"],
        errors="coerce"
    )

    nba_positions = nba_positions.dropna(
        subset=["PLAYER_ID"]
    )

    nba_positions["PLAYER_ID"] = (
        nba_positions["PLAYER_ID"].astype(int)
    )

    # Remove duplicate player IDs.
    nba_positions = nba_positions.drop_duplicates(
        subset=["PLAYER_ID"],
        keep="last"
    )

    # ---------------------------------------------------------
    # 7. Match NBA positions to the players in our box scores
    # ---------------------------------------------------------

    current_players = players.merge(
        nba_positions,
        on="PLAYER_ID",
        how="left"
    )

    # ---------------------------------------------------------
    # 8. Merge with existing cache
    #
    # New NBA data wins when a position is available.
    # Existing cached positions are retained otherwise.
    # ---------------------------------------------------------

    current_players = current_players.rename(
        columns={
            "POSITION": "NEW_POSITION"
        }
    )

    existing_lookup = existing[
        ["PLAYER_ID", "POSITION"]
    ].rename(
        columns={
            "POSITION": "CACHED_POSITION"
        }
    )

    combined = current_players.merge(
        existing_lookup,
        on="PLAYER_ID",
        how="left"
    )

    # Prefer the freshly fetched NBA position.
    combined["POSITION"] = (
        combined["NEW_POSITION"]
        .fillna(combined["CACHED_POSITION"])
        .fillna("Unknown")
    )

    # ---------------------------------------------------------
    # 9. Keep only the columns we need
    # ---------------------------------------------------------

    combined = combined[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "POSITION",
        ]
    ]

    combined = combined.drop_duplicates(
        subset=["PLAYER_ID"],
        keep="last"
    )

    combined = combined.sort_values(
        ["PLAYER_NAME", "PLAYER_ID"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 10. Report results
    # ---------------------------------------------------------

    unknown = combined[
        combined["POSITION"].eq("Unknown")
    ]

    print(
        f"Players saved: {len(combined)}"
    )

    print(
        f"Players with positions: "
        f"{len(combined) - len(unknown)}"
    )

    if not unknown.empty:

        print(
            f"⚠️ Players with unknown positions: "
            f"{len(unknown)}"
        )

        for _, row in unknown.iterrows():

            print(
                f"    {row['PLAYER_NAME']} "
                f"({row['PLAYER_ID']})"
            )

    # ---------------------------------------------------------
    # 11. Save local position cache
    #
    # This is the ONLY file this function writes.
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    combined.to_csv(
        output_path,
        index=False
    )

    print(
        f"✅ Saved player positions to:\n"
        f"   {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch/update NBA player box scores.")
    parser.add_argument("--full-refresh", action="store_true",
                         help="Force a full historical re-backfill, ignoring any existing store.")
    args = parser.parse_args()
    update_boxscores(force_full_refresh=args.full_refresh)
    update_player_positions_cache(DATA_DIRECTORY)