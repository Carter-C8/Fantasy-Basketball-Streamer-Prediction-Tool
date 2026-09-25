# import os
# from fastapi import FastAPI, Header, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from espn_api.basketball import League
# import pandas as pd
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# # Browsers block a page on one origin (localhost:5173) from fetching data
# # from a different origin (localhost:8000) unless the server explicitly
# # says it's allowed. This is a BROWSER security rule, not a Python one --
# # curl or a Python script calling this API would never hit this at all.
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Absolute path derived from this file's own location, not a relative
# # "../data_pipeline/..." string -- the relative version only works if you
# # happen to launch uvicorn from inside server/ specifically. This is the
# # exact same __file__-based fix already applied throughout the rest of
# # this project, for the same reason.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# WEEKLY_TOTALS_PATH = os.path.join(
#     PROJECT_ROOT, "data_pipeline", "nba_live_data", "predictions", "weekly_totals.csv"
# )

# @app.get("/api/streamers/weekly")
# def get_streamers(
#     league_id: int = None,
#     week_offset: int = 0,  # 0 = Current Week, 1 = Next Week
#     x_espn_s2: str = Header(None),
#     x_espn_swid: str = Header(None)
# ):
#     csv_path = "../data_pipeline/nba_live_data/predictions/weekly_totals.csv"
#     if not os.path.exists(csv_path):
#         raise HTTPException(status_code=404, detail="Predictions not found.")
        
#     weekly_preds = pd.read_csv(csv_path).fillna("")
    
#     # Figure out the target ISO week based on the offset
#     # target_date = pd.Timestamp.today() + pd.Timedelta(days=7 * week_offset)
#     base_date = pd.Timestamp('2026-10-20')
#     target_date = base_date + pd.Timedelta(days=7 * week_offset)
#     target_year, target_week = target_date.isocalendar().year, target_date.isocalendar().week
    
#     # Filter the CSV down to ONLY the requested week
#     weekly_preds = weekly_preds[
#         (weekly_preds['ISO_YEAR'] == target_year) & 
#         (weekly_preds['ISO_WEEK'] == target_week)
#     ]
    
#     # ... rest of your existing ESPN filtering logic ...
#     return weekly_preds.to_dict(orient="records")

# @app.get("/api/player/{player_name}")
# def get_player_details(player_name: str, week_offset: int = 0): # <-- Added week_offset
#     # 1. Load the un-aggregated full week of games
#     weekly_games = pd.read_csv("../data_pipeline/nba_live_data/predictions/weekly_individual_games.csv")
    
#     # Calculate target ISO year and week based on the offset
#     # target_date = pd.Timestamp.today() + pd.Timedelta(days=7 * week_offset)
#     base_date = pd.Timestamp('2026-10-20')
#     target_date = base_date + pd.Timedelta(days=7 * week_offset)
#     target_year, target_week = target_date.isocalendar().year, target_date.isocalendar().week

#     # 2. Filter for this player AND the specific ISO week
#     player_games = weekly_games[
#         (weekly_games['PLAYER_NAME'] == player_name) &
#         (weekly_games['ISO_YEAR'] == target_year) &
#         (weekly_games['ISO_WEEK'] == target_week)
#     ]
    
#     # Convert timestamp to string YYYY-MM-DD so JSON can serialize it
#     player_games = player_games.copy()
#     player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
#     # 3. Load historical 10-game average
#     historical_data = pd.read_csv("../data_pipeline/nba_live_data/processed/player_features.csv")
#     player_history = historical_data[historical_data['PLAYER_NAME'] == player_name]
#     avg_last_10 = player_history['FP_ROLLING_10'].iloc[-1] if not player_history.empty else 0
    
#     # 4. Format upcoming games
#     pred_col = 'FINAL_PREDICTION' if 'FINAL_PREDICTION' in player_games.columns else 'ML_PREDICTION'
#     upcoming_schedule = player_games[['GAME_DATE', pred_col]].rename(
#         columns={pred_col: 'ML_PREDICTION'}
#     ).to_dict(orient="records")
    
#     return {
#         "player_name": player_name,
#         "actual_10_game_avg": float(avg_last_10),
#         "upcoming_games": upcoming_schedule
#     }
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from espn_api.basketball import League
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data_pipeline", "nba_live_data")

WEEKLY_TOTALS_PATH = os.path.join(DATA_DIRECTORY, "predictions", "weekly_totals.csv")
WEEKLY_INDIVIDUAL_GAMES_PATH = os.path.join(DATA_DIRECTORY, "predictions", "weekly_individual_games.csv")
PLAYER_FEATURES_PATH = os.path.join(DATA_DIRECTORY, "processed", "player_features.csv")

# NEW: Path to your locally generated static positions CSV
PLAYER_POSITIONS_PATH = os.path.join(DATA_DIRECTORY, "player_positions.csv")


def current_season_start_year(today) -> int:
    return today.year if today.month >= 8 else today.year - 1


def get_target_iso_week(week_offset: int):
    # TODO: this is still pinned to a fixed testing date (left over from
    # the version this file was built from) instead of the real current
    # date -- swap to pd.Timestamp.today() before relying on this for
    # actual live use, or Current/Next Week will always mean the same
    # two fixed calendar weeks regardless of when it's actually run.
    base_date = pd.Timestamp('2026-10-20')
    target_date = base_date + pd.Timedelta(days=7 * week_offset)
    iso = target_date.isocalendar()
    return iso.year, iso.week


# ---------------------------------------------------------------------
# CHANGE 4: Historical top-156 filter
# ---------------------------------------------------------------------
# Mirrors label_streamers' PRIOR-SEASON RANK logic from
# preprocess_nba_data.ipynb: a player counts as a locked-in top player
# for the CURRENT season if their average FANTASY_PTS in the PRIOR
# season put them in the top `top_n` league-wide. This is deliberately
# NOT "top 156 of this week's predictions" -- it's the same historical,
# whole-prior-season definition used when the training data was built,
# applied here to whichever season is current right now.
_prior_season_top156_cache = None


def compute_prior_season_top_players(top_n: int = 156) -> set:
    if not os.path.exists(PLAYER_FEATURES_PATH):
        raise HTTPException(status_code=404, detail="Processed feature store not found -- run the data pipeline first.")

    df = pd.read_csv(PLAYER_FEATURES_PATH, parse_dates=['GAME_DATE'],
                      usecols=['PLAYER_ID', 'GAME_DATE', 'FANTASY_PTS'])
    
    # 1. Assign season to each game
    df['SEASON'] = np.where(df['GAME_DATE'].dt.month >= 8, df['GAME_DATE'].dt.year, df['GAME_DATE'].dt.year - 1)

    # 2. Determine what the "prior season" is. 
    # (If today is Sept 2026, current_season is 2026. Therefore prior_season is 2025).
    current_season = current_season_start_year(pd.Timestamp.today())
    prior_season = current_season - 1

    # 3. Filter to ONLY the prior season's games
    prior_season_data = df[df['SEASON'] == prior_season]
    
    if prior_season_data.empty:
        # Fallback if no prior season data exists (e.g., brand new database)
        return set()

    # 4. Calculate average fantasy points per player and rank them
    season_avg = prior_season_data.groupby('PLAYER_ID')['FANTASY_PTS'].mean().reset_index()
    season_avg['PRIOR_SEASON_RANK'] = season_avg['FANTASY_PTS'].rank(ascending=False, method='first')

    # 5. Return the IDs of the top 156 players
    return set(season_avg.loc[season_avg['PRIOR_SEASON_RANK'] <= top_n, 'PLAYER_ID'])


def get_prior_season_top156() -> set:
    # Cached in memory for the life of the server process -- prior-season
    # rank can't change mid-season, so there's no reason to recompute it
    # on every request. (Using --reload in dev will reset this cache
    # whenever the file changes, which is fine for local development.)
    global _prior_season_top156_cache
    if _prior_season_top156_cache is None:
        _prior_season_top156_cache = compute_prior_season_top_players(top_n=156)
    return _prior_season_top156_cache


FANTASY_TO_NBA_POSITION = {
    'GUARD': 'G',
    'FORWARD': 'F',
    'CENTER': 'C',
    # Fallback support just in case the frontend still sends these
    'PG': 'G', 'SG': 'G',
    'SF': 'F', 'PF': 'F',
    'C': 'C',
}

@app.get("/api/streamers/weekly")
def get_streamers(
    league_id: int = None,
    week_offset: int = 0,
    position: Optional[str] = None,
    hide_top_156: bool = False,
    x_espn_s2: str = Header(None),
    x_espn_swid: str = Header(None),
):
    if not os.path.exists(WEEKLY_TOTALS_PATH):
        raise HTTPException(status_code=404, detail="Predictions not found.")

    weekly_preds = pd.read_csv(WEEKLY_TOTALS_PATH).fillna("")

    NBA_TEAMS = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
    }

    # Add TEAM_ABBREV to the output
    weekly_preds['TEAM_ABBREV'] = weekly_preds['TEAM_ID'].map(NBA_TEAMS).fillna('')

    target_year, target_week = get_target_iso_week(week_offset)
    weekly_preds = weekly_preds[
        (weekly_preds['ISO_YEAR'] == target_year) &
        (weekly_preds['ISO_WEEK'] == target_week)
    ]

    # --- Historical top-156 filter ---
    if hide_top_156:
        top156_ids = get_prior_season_top156()
        weekly_preds = weekly_preds[~weekly_preds['PLAYER_ID'].isin(top156_ids)]

    # --- Local CSV Position filter ---
    if os.path.exists(PLAYER_POSITIONS_PATH):
            positions_df = pd.read_csv(PLAYER_POSITIONS_PATH)
            if 'POSITION' in weekly_preds.columns:
                weekly_preds = weekly_preds.drop(columns=['POSITION'])
            weekly_preds = weekly_preds.merge(
                positions_df[['PLAYER_ID', 'POSITION']],
                on='PLAYER_ID',
                how='left'
            )
            weekly_preds['POSITION'] = weekly_preds['POSITION'].fillna('—')

        # --- Filter by position if one is selected ---
    if position:
        nba_position = FANTASY_TO_NBA_POSITION.get(position.upper())
        if nba_position:
            weekly_preds = weekly_preds[
                weekly_preds['POSITION'].str.contains(nba_position, na=False)
            ]

    # --- ESPN free-agent filtering ---
    if league_id and x_espn_s2 and x_espn_swid:
        try:
            league = League(league_id=league_id, year=2026, espn_s2=x_espn_s2, swid=x_espn_swid)
            free_agents = league.free_agents(size=150)
            available_names = {player.name for player in free_agents}
            weekly_preds = weekly_preds[weekly_preds['PLAYER_NAME'].isin(available_names)]
        except Exception as e:
            print(f"ESPN API Error: {e}")

    return weekly_preds.to_dict(orient="records")


@app.get("/api/player/{player_name}")
def get_player_details(player_name: str, week_offset: int = 0):
    if not os.path.exists(WEEKLY_INDIVIDUAL_GAMES_PATH):
        raise HTTPException(status_code=404, detail="Weekly individual games file not found.")
    
    weekly_games = pd.read_csv(WEEKLY_INDIVIDUAL_GAMES_PATH)
    target_year, target_week = get_target_iso_week(week_offset)

    player_games = weekly_games[
        (weekly_games['PLAYER_NAME'] == player_name) &
        (weekly_games['ISO_YEAR'] == target_year) &
        (weekly_games['ISO_WEEK'] == target_week)
    ].copy()

    player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE']).dt.strftime('%Y-%m-%d')

    if not os.path.exists(PLAYER_FEATURES_PATH):
        raise HTTPException(status_code=404, detail="Processed feature store not found.")
    
    historical_data = pd.read_csv(PLAYER_FEATURES_PATH)
    player_history = historical_data[historical_data['PLAYER_NAME'] == player_name]
    
    if player_history.empty:
        raise HTTPException(status_code=404, detail="Player not found in historical data.")
        
    # Get the player ID for the headshot and their most recent rolling averages
    latest_stats = player_history.iloc[-1]
    player_id = int(latest_stats['PLAYER_ID'])
    avg_last_10 = float(latest_stats.get('FP_ROLLING_10', 0))
    
    # Bundle the latest rolling baseline stats to display in the dropdown
    baseline_stats = {
        "MIN": float(latest_stats.get('MIN_ROLLING_4', 0)),
        "PTS": float(latest_stats.get('PTS_ROLLING_4', 0)),
        "REB": float(latest_stats.get('REB_ROLLING_4', 0)),
        "AST": float(latest_stats.get('AST_ROLLING_4', 0)),
        "STL": float(latest_stats.get('STL_ROLLING_4', 0)),
        "BLK": float(latest_stats.get('BLK_ROLLING_4', 0))
    }

    pred_col = 'FINAL_PREDICTION' if 'FINAL_PREDICTION' in player_games.columns else 'ML_PREDICTION'
    

    
    # Dictionary mapping NBA numeric IDs to 3-letter abbreviations
    NBA_TEAMS = {
        '1610612737': 'ATL', '1610612738': 'BOS', '1610612751': 'BKN', '1610612766': 'CHA',
        '1610612741': 'CHI', '1610612739': 'CLE', '1610612742': 'DAL', '1610612743': 'DEN',
        '1610612765': 'DET', '1610612744': 'GSW', '1610612745': 'HOU', '1610612754': 'IND',
        '1610612746': 'LAC', '1610612747': 'LAL', '1610612763': 'MEM', '1610612748': 'MIA',
        '1610612749': 'MIL', '1610612750': 'MIN', '1610612740': 'NOP', '1610612752': 'NYK',
        '1610612760': 'OKC', '1610612753': 'ORL', '1610612755': 'PHI', '1610612756': 'PHX',
        '1610612757': 'POR', '1610612758': 'SAC', '1610612759': 'SAS', '1610612761': 'TOR',
        '1610612762': 'UTA', '1610612764': 'WAS'
    }

    # Include mapped OPPONENT in the payload
    upcoming_schedule = []
    if not player_games.empty:
        for _, row in player_games.iterrows():
            raw_id = str(row.get('OPPONENT_TEAM_ID', '')).replace('.0', '')
            
            # If the ID is already a 3-letter abbreviation (e.g., 'CLE'), keep it.
            # Otherwise, use the dictionary to convert the numeric ID.
            if len(raw_id) == 3 and raw_id.isalpha():
                mapped_opponent = raw_id
            else:
                mapped_opponent = NBA_TEAMS.get(raw_id, 'UNK')
            
            upcoming_schedule.append({
                "GAME_DATE": row['GAME_DATE'],
                "OPPONENT": mapped_opponent,
                "ML_PREDICTION": float(row[pred_col])
            })

    return {
        "player_id": player_id,
        "player_name": player_name,
        "actual_10_game_avg": avg_last_10,
        "baseline_stats": baseline_stats,
        "upcoming_games": upcoming_schedule
    }