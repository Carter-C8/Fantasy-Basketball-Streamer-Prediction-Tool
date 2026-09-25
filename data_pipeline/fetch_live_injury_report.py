"""
fetch_injury_report.py

Phase 2 data script: pulls confirmed player injury/participation status
for a given game date using the `nbainjuries` package, which handles
retrieving and parsing the NBA's official injury report snapshots
directly (see https://github.com/mxufc29/nbainjuries) -- no manual PDF
scraping needed here, unlike an earlier draft of this script.

Requires:
  - pip install nbainjuries        (needs Python 3.10+)
  - A Java Runtime (JRE 8+) on PATH -- nbainjuries uses tabula-py under
    the hood to read the report PDFs. Verify with `java -version` before
    relying on this. This also matters for deployment: a standard
    "pip install -r requirements.txt" buildpack likely won't include a
    JRE, so Render/Railway may need a custom Docker image with Java
    installed rather than the default Python buildpack.

nbainjuries' own documented example queries a same-day 5:30pm ET
snapshot and gets back that same day's evening games -- this script
mirrors that pattern by default. Report data updates throughout the day;
if you need an earlier snapshot (e.g. for the second game of a
back-to-back, which teams must report on by 1pm local time same-day),
pass --hour/--minute to query an earlier snapshot.

Player names come back as "Last, First" and teams as full team names
(e.g. "Boston Celtics") -- both get mapped here to the PLAYER_ID /
TEAM_ID the rest of this pipeline already uses: team names via nba_api's
static team list, player names via the box score history from Phase 1.

Usage:
    python fetch_injury_report.py                   # today, 5:30pm ET snapshot
    python fetch_injury_report.py --date 2026-01-15
    python fetch_injury_report.py --hour 13 --minute 0  # 1pm snapshot, e.g. for a back-to-back
"""

import argparse
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
from nbainjuries import injury
from nba_api.stats.static import teams as nba_teams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "nba_live_data")
INJURY_DIRECTORY = os.path.join(DATA_DIRECTORY, "injuries")
BOXSCORE_PATH = os.path.join(DATA_DIRECTORY, "scores", "player_boxscores.csv")

REPORT_HOUR = 17    # 5:30pm ET, matching nbainjuries' own documented example
REPORT_MINUTE = 30

OUT_STATUSES = {"Out", "Doubtful"}  # not expected to play; Questionable/Probable/Available left active


def _team_name_lookup() -> dict:
    """Full team name (e.g. 'Boston Celtics') -> TEAM_ID."""
    return {t["full_name"]: t["id"] for t in nba_teams.get_teams()}


def _normalize_player_name(name: str) -> str:
    """'Brown, Jaylen' -> 'jaylen brown', to match box-score PLAYER_NAME format."""
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}"
    return name.strip().lower()


def _build_player_name_lookup() -> dict:
    if not os.path.exists(BOXSCORE_PATH):
        raise FileNotFoundError(
            f"No box score history at {BOXSCORE_PATH}. Run fetch_boxscores.py first -- "
            f"name matching needs it to resolve PLAYER_NAME -> PLAYER_ID."
        )
    boxscores = pd.read_csv(BOXSCORE_PATH, usecols=["PLAYER_ID", "PLAYER_NAME"])
    boxscores = boxscores.drop_duplicates(subset="PLAYER_NAME", keep="last")
    return {row.PLAYER_NAME.strip().lower(): row.PLAYER_ID for row in boxscores.itertuples()}


def fetch_injury_report(report_datetime: datetime) -> pd.DataFrame:
    print(f"Fetching injury report snapshot for {report_datetime.strftime('%Y-%m-%d %I:%M %p')}...")

    try:
        report_df = injury.get_reportdata(report_datetime, return_df=True)
    except Exception as e:
        print(f"  nbainjuries raised an error fetching this snapshot: {e}")
        print("  (No report may exist for this exact date/time -- try a different --hour/--minute, "
              "or check whether games are even scheduled for this date.)")
        report_df = None

    if report_df is None or report_df.empty:
        print("  No report data returned.")
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME",
                                      "STATUS", "REASON", "IS_OUT", "GAME_DATE"])

    team_lookup = _team_name_lookup()
    player_lookup = _build_player_name_lookup()

    report_df = report_df.rename(columns={
        "Player Name": "PLAYER_NAME_RAW",
        "Team": "TEAM_NAME",
        "Current Status": "STATUS",
        "Reason": "REASON",
        "Game Date": "GAME_DATE",
    })

    report_df["PLAYER_NAME"] = report_df["PLAYER_NAME_RAW"].apply(_normalize_player_name)
    report_df["PLAYER_ID"] = report_df["PLAYER_NAME"].map(player_lookup)
    report_df["TEAM_ID"] = report_df["TEAM_NAME"].map(team_lookup)

    unmatched_players = report_df[report_df["PLAYER_ID"].isna()]
    if len(unmatched_players) > 0:
        print(f"  Warning: {len(unmatched_players)} player name(s) didn't match box score history "
              f"(rookies not yet on record, nicknames, etc.) -- logged, not dropped:")
        for name in unmatched_players["PLAYER_NAME_RAW"].unique():
            print(f"    - {name}")

    unmatched_teams = report_df[report_df["TEAM_ID"].isna()]
    if len(unmatched_teams) > 0:
        print(f"  Warning: {len(unmatched_teams)} team name(s) didn't match nba_api's team list: "
              f"{unmatched_teams['TEAM_NAME'].unique().tolist()}")

    report_df["GAME_DATE"] = pd.to_datetime(report_df["GAME_DATE"], format="%m/%d/%Y")
    report_df["IS_OUT"] = report_df["STATUS"].isin(OUT_STATUSES)

    return report_df[["PLAYER_ID", "PLAYER_NAME_RAW", "TEAM_ID", "TEAM_NAME",
                       "STATUS", "REASON", "IS_OUT", "GAME_DATE"]].rename(
        columns={"PLAYER_NAME_RAW": "PLAYER_NAME"}
    )


def save_injury_report(report: pd.DataFrame, game_date: str) -> str:
    os.makedirs(INJURY_DIRECTORY, exist_ok=True)
    output_path = os.path.join(INJURY_DIRECTORY, f"injuries_{game_date}.csv")
    report.to_csv(output_path, index=False)
    return output_path


def update_injury_report(game_date: Optional[str] = None,
                          hour: int = REPORT_HOUR, minute: int = REPORT_MINUTE) -> pd.DataFrame:
    target_date = datetime.strptime(game_date, "%Y-%m-%d").date() if game_date else date.today()
    report_datetime = datetime(target_date.year, target_date.month, target_date.day, hour, minute)

    report = fetch_injury_report(report_datetime)
    output_path = save_injury_report(report, target_date.strftime("%Y-%m-%d"))
    print(f"{len(report)} player statuses ({int(report['IS_OUT'].sum())} confirmed out/doubtful) "
          f"saved to {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the NBA's injury report via the nbainjuries package.")
    parser.add_argument("--date", type=str, default=None,
                         help="Game date to fetch the report for, 'YYYY-MM-DD'. Defaults to today.")
    parser.add_argument("--hour", type=int, default=REPORT_HOUR, help="Report snapshot hour (ET, 24h).")
    parser.add_argument("--minute", type=int, default=REPORT_MINUTE, help="Report snapshot minute.")
    args = parser.parse_args()
    update_injury_report(args.date, hour=args.hour, minute=args.minute)