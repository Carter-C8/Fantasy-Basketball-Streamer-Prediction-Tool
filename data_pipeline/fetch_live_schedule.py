"""
fetch_schedule.py

Phase 2 data script: pulls tonight's (or any given date's) NBA schedule --
which teams play, and who's home/away. None of this exists in a box score
until after the game is played, so it can't come from Phase 1's data.

This is what lets predict_today.py build a feature row for a player whose
game hasn't happened yet: OPPONENT_TEAM_ID and HOME_GAME, the same two
things clean_data() currently derives from MATCHUP text after the fact --
produced here instead, before tip-off.

Output: one row per (TEAM_ID, GAME_DATE) for every team playing that date,
written to nba_data/schedule/schedule_<date>.csv. Always overwritten on
re-run for the same date -- a day's schedule doesn't change once games
are set, so there's no "incremental" concept needed here the way there is
for box scores.

NOTE: written against nba_api's documented ScoreboardV2 interface but not
verified against a live network call from this environment -- run it once
and sanity-check the output before wiring it into anything downstream.

Usage:
    python fetch_schedule.py                   # today's schedule
    python fetch_schedule.py --date 2026-01-15  # a specific date (e.g. for testing)
"""

import argparse
import os
from datetime import date
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import scoreboardv3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "nba_live_data")
SCHEDULE_DIRECTORY = os.path.join(DATA_DIRECTORY, "schedule")


def fetch_schedule_for_date(game_date: str) -> pd.DataFrame:
    """game_date: 'YYYY-MM-DD' string."""
    board = scoreboardv3.ScoreboardV3(game_date=game_date)
    
    # Extract the raw dictionary instead of relying on fragile dataframes
    data = board.get_dict()
    games = data.get('scoreboard', {}).get('games', [])

    if not games:
        print(f"  No games scheduled for {game_date}.")
        return pd.DataFrame(columns=["TEAM_ID", "OPPONENT_TEAM_ID", "GAME_DATE", "GAME_ID", "HOME_GAME"])

    rows = []
    for game in games:
        game_id = game.get('gameId')
        home_team_id = game.get('homeTeam', {}).get('teamId')
        away_team_id = game.get('awayTeam', {}).get('teamId')
        
        # Build the Home row
        rows.append({
            "TEAM_ID": home_team_id,
            "OPPONENT_TEAM_ID": away_team_id,
            "GAME_DATE": game_date,
            "GAME_ID": game_id,
            "HOME_GAME": 1
        })
        # Build the Away row
        rows.append({
            "TEAM_ID": away_team_id,
            "OPPONENT_TEAM_ID": home_team_id,
            "GAME_DATE": game_date,
            "GAME_ID": game_id,
            "HOME_GAME": 0
        })

    schedule = pd.DataFrame(rows)
    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"])
    return schedule[["TEAM_ID", "OPPONENT_TEAM_ID", "GAME_DATE", "GAME_ID", "HOME_GAME"]]


def save_schedule(schedule: pd.DataFrame, game_date: str) -> str:
    os.makedirs(SCHEDULE_DIRECTORY, exist_ok=True)
    output_path = os.path.join(SCHEDULE_DIRECTORY, f"schedule_{game_date}.csv")
    schedule.to_csv(output_path, index=False)
    return output_path


def update_schedule(game_date: Optional[str] = None) -> pd.DataFrame:
    game_date = game_date or date.today().strftime("%Y-%m-%d")
    print(f"Fetching schedule for {game_date}...")
    schedule = fetch_schedule_for_date(game_date)
    output_path = save_schedule(schedule, game_date)
    print(f"{len(schedule)} team-rows ({len(schedule) // 2} games) saved to {output_path}")
    return schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the NBA schedule for a given date.")
    parser.add_argument("--date", type=str, default=None,
                         help="Date to fetch, 'YYYY-MM-DD'. Defaults to today.")
    args = parser.parse_args()
    update_schedule(args.date)