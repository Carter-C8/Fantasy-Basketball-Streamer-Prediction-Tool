"""
predict_today.py

Phase 4 script: loads the trained model and today's live data, and
produces a ranked prediction for every streamer-eligible player playing
today. Output is appended to a running predictions log with one row per
(PLAYER_ID, GAME_DATE) -- the exact shape a later weekly-aggregation
script (Phase 6) needs to sum a player's predictions across a week.

Assembles tonight's feature row for each candidate player from three
already-built sources:
  - their most recent rolling-stat row from the processed feature store
    (preprocess_live_data.py's output) -- FP_ROLLING_*, MIN_ROLLING_*, etc.
  - tonight's OPPONENT_TEAM_ID / HOME_GAME from the schedule
    (fetch_live_schedule.py's output)
  - tonight's TEAM_VACATED_* stats, recomputed HERE using the injury
    report's expected-active roster (fetch_live_injury_report.py's
    output) instead of "who's in tonight's box score" -- which doesn't
    exist yet, pre-game.

FILTER ORDERING MATTERS:
  - Confirmed OUT/Doubtful players are dropped BEFORE the vacated-stats
    calculation -- they're not on the floor, so they shouldn't count
    toward their team's active roster.
  - Players under MIN_SEASON_GAMES games played this season are dropped
    AFTER vacated stats are computed, and only from the final ranked
    output -- a low-season-game player (a recent trade addition, e.g.)
    is still really occupying minutes, so excluding them earlier would
    make their team look like it has more open opportunity than it does.
    This filter uses SEASON games, not career games, so it catches both
    rookies (season games this year) and players back from a long
    injury (few recent games despite a long career).

Requires model.pkl and a matching feature list to already exist (Phase 3,
train_model.py) -- this script loads them, it doesn't train anything.

Usage:
    python predict_today.py
    python predict_today.py --date 2026-01-15
"""

import argparse
import json
import os
from datetime import date
from typing import Optional

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../NBA_STREAMER_ML_MODEL/live_inference_scripts
NBA_STREAMER_ML_MODEL_DIR = os.path.dirname(SCRIPT_DIR)           # .../NBA_STREAMER_ML_MODEL
PROJECT_ROOT = os.path.dirname(NBA_STREAMER_ML_MODEL_DIR)         # .../NBA_Fantasy_Basketball_Project

DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data_pipeline", "nba_live_data")
PROCESSED_PATH = os.path.join(DATA_DIRECTORY, "processed", "player_features.csv")
SCHEDULE_DIRECTORY = os.path.join(DATA_DIRECTORY, "schedule")
INJURY_DIRECTORY = os.path.join(DATA_DIRECTORY, "injuries")
PREDICTIONS_PATH = os.path.join(DATA_DIRECTORY, "predictions", "daily_predictions.csv")

MODEL_DIRECTORY = os.path.join(NBA_STREAMER_ML_MODEL_DIR, "ml_engine")
MODEL_PATH = os.path.join(MODEL_DIRECTORY, "nba_streamer_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIRECTORY, "features.json")

# Fallback if features.json isn't found -- MUST exactly match what
# train_model.py actually trained on. Prefer the persisted file; this
# only exists so the script can still run/be tested without it.
DEFAULT_FEATURES = [
    'FP_ROLLING_3', 'FP_ROLLING_4', 'FP_ROLLING_10', 'FP_PER_MIN_10',
    'MIN_ROLLING_3', 'MIN_ROLLING_4', 'MIN_ROLLING_10',
    'PTS_ROLLING_4', 'REB_ROLLING_4', 'AST_ROLLING_4',
    'STL_ROLLING_4', 'BLK_ROLLING_4', 'TOV_ROLLING_4', 'FG3M_ROLLING_4',
    'TEAM_VACATED_MINUTES', 'TEAM_VACATED_PTS',
    'TEAM_VACATED_REB', 'TEAM_VACATED_AST',
    'HOME_GAME',
]

MIN_SEASON_GAMES = 0


def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train_model.py first (Phase 3).")
    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            features = json.load(f)
    else:
        print(f"  Warning: no {FEATURES_PATH} found -- falling back to the hardcoded default feature "
              f"list. Verify this matches what the model was actually trained on.")
        features = DEFAULT_FEATURES
    return model, features


def current_season_start_year(today: date) -> int:
    return today.year if today.month >= 8 else today.year - 1


def load_latest_player_state(processed: pd.DataFrame) -> pd.DataFrame:
    """One row per player: their most recent completed game's rolling
    stats, which is the lookback state a fresh prediction is built from."""
    latest_idx = processed.groupby('PLAYER_ID')['GAME_DATE'].idxmax()
    latest = processed.loc[latest_idx].reset_index(drop=True)
    # HOME_GAME/OPPONENT here reflect that player's LAST completed game,
    # not tonight's -- drop them so tonight's schedule data (merged next)
    # is unambiguously what gets used, with no risk of a merge collision
    # silently producing HOME_GAME_x/HOME_GAME_y instead of HOME_GAME.
    return latest.drop(columns=['HOME_GAME', 'OPPONENT'], errors='ignore')


def count_season_games(processed: pd.DataFrame, season_start_year: int) -> pd.Series:
    season_of_row = np.where(
        processed['GAME_DATE'].dt.month >= 8,
        processed['GAME_DATE'].dt.year,
        processed['GAME_DATE'].dt.year - 1,
    )
    season_rows = processed[season_of_row == season_start_year]
    return season_rows.groupby('PLAYER_ID').size().rename('SEASON_GAMES_PLAYED')


def build_expected_active_roster(processed: pd.DataFrame, schedule: pd.DataFrame,
                                  injuries: Optional[pd.DataFrame],
                                  season_start_year: int, game_date: pd.Timestamp,
                                  latest_state: Optional[pd.DataFrame] = None,
                                  season_games: Optional[pd.Series] = None) -> pd.DataFrame:
    """Everyone on a team playing tonight, minus confirmed Out/Doubtful
    players. This is the roster used for BOTH the vacated-stats
    calculation and (after a further games-played filter) the final
    prediction pool.

    latest_state/season_games can be precomputed and passed in when this
    is called repeatedly against the same `processed` snapshot for
    multiple dates (aggregate_weekly.py's full-week forecast does this,
    once per remaining day in the week) -- avoids redundantly recomputing
    the same per-player lookback on every call. A normal single-day run
    (this file's own run_daily_prediction) only calls this once, so it
    leaves both as None and they're computed fresh, as before.
    """
    if latest_state is None:
        latest_state = load_latest_player_state(processed)
    if season_games is None:
        season_games = count_season_games(processed, season_start_year)
    latest_state = latest_state.merge(season_games, on='PLAYER_ID', how='left')
    latest_state['SEASON_GAMES_PLAYED'] = latest_state['SEASON_GAMES_PLAYED'].fillna(0)

    expected_active = latest_state.merge(
        schedule[['TEAM_ID', 'OPPONENT_TEAM_ID', 'HOME_GAME']], on='TEAM_ID', how='inner'
    )

    if injuries is not None and not injuries.empty:
        out_ids = set(injuries.loc[injuries['IS_OUT'], 'PLAYER_ID'].dropna())
        before = len(expected_active)
        expected_active = expected_active[~expected_active['PLAYER_ID'].isin(out_ids)]
        print(f"  Dropped {before - len(expected_active)} players confirmed Out/Doubtful.")
    else:
        print("  Warning: no injury report loaded -- predictions won't reflect tonight's confirmed absences.")

    expected_active = expected_active.copy()
    expected_active['GAME_DATE'] = game_date
    return expected_active


def recompute_vacated_stats_for_tonight(expected_active: pd.DataFrame) -> pd.DataFrame:
    """Same logic as add_vacated_stats() in the historical pipeline, but
    using tonight's EXPECTED active roster instead of who ends up in a
    box score that doesn't exist yet."""
    df = expected_active.copy()
    df['EXPECTED_MIN'] = df['MIN_ROLLING_4']
    df['EXPECTED_PTS'] = df['PTS_ROLLING_4']
    df['EXPECTED_REB'] = df['REB_ROLLING_4']
    df['EXPECTED_AST'] = df['AST_ROLLING_4']

    team_expected = df.groupby('TEAM_ID')[['EXPECTED_MIN', 'EXPECTED_PTS', 'EXPECTED_REB', 'EXPECTED_AST']].sum()
    team_expected = team_expected.rename(columns={
        'EXPECTED_MIN': 'TEAM_ACTIVE_MIN', 'EXPECTED_PTS': 'TEAM_ACTIVE_PTS',
        'EXPECTED_REB': 'TEAM_ACTIVE_REB', 'EXPECTED_AST': 'TEAM_ACTIVE_AST',
    })

    df = df.merge(team_expected, on='TEAM_ID', how='left')
    df['TEAM_VACATED_MINUTES'] = (240.0 - df['TEAM_ACTIVE_MIN']).clip(lower=0)
    df['TEAM_VACATED_PTS'] = (115.0 - df['TEAM_ACTIVE_PTS']).clip(lower=0)
    df['TEAM_VACATED_REB'] = (45.0 - df['TEAM_ACTIVE_REB']).clip(lower=0)
    df['TEAM_VACATED_AST'] = (25.0 - df['TEAM_ACTIVE_AST']).clip(lower=0)

    return df.drop(columns=['EXPECTED_MIN', 'EXPECTED_PTS', 'EXPECTED_REB', 'EXPECTED_AST',
                             'TEAM_ACTIVE_MIN', 'TEAM_ACTIVE_PTS', 'TEAM_ACTIVE_REB', 'TEAM_ACTIVE_AST'])

def add_live_opponent_metrics(rankable_df, historical_features_df):
    df = rankable_df.copy()
    
    # Map numeric NBA team IDs to 3-letter abbreviation strings
    id_to_abbrev = {
        '1610612737': 'ATL', '1610612738': 'BOS', '1610612751': 'BKN', '1610612766': 'CHA',
        '1610612741': 'CHI', '1610612739': 'CLE', '1610612742': 'DAL', '1610612743': 'DEN',
        '1610612765': 'DET', '1610612744': 'GSW', '1610612745': 'HOU', '1610612754': 'IND',
        '1610612746': 'LAC', '1610612747': 'LAL', '1610612763': 'MEM', '1610612748': 'MIA',
        '1610612749': 'MIL', '1610612750': 'MIN', '1610612740': 'NOP', '1610612752': 'NYK',
        '1610612760': 'OKC', '1610612753': 'ORL', '1610612755': 'PHI', '1610612756': 'PHX',
        '1610612757': 'POR', '1610612758': 'SAC', '1610612759': 'SAS', '1610612761': 'TOR',
        '1610612762': 'UTA', '1610612764': 'WAS'
    }
    
    # Convert numeric ID to abbreviation, keeping original string if mapping fails
    df['OPPONENT_TEAM_ID'] = df['OPPONENT_TEAM_ID'].astype(str).map(id_to_abbrev).fillna(df['OPPONENT_TEAM_ID'])
    
    latest_defense = historical_features_df.dropna(subset=['OPP_DEF_RTG_10']).sort_values('GAME_DATE').groupby('OPPONENT').last().reset_index()

    team_defense_lookup = latest_defense[['OPPONENT', 'OPP_DEF_RTG_10', 'OPP_PACE_10']].rename(
        columns={'OPPONENT': 'OPPONENT_TEAM_ID'}
    )
    team_defense_lookup['OPPONENT_TEAM_ID'] = team_defense_lookup['OPPONENT_TEAM_ID'].astype(str)

    df = df.merge(
        team_defense_lookup,
        on='OPPONENT_TEAM_ID',
        how='left'
    )

    if 'OPP_DEF_RTG_10' not in df.columns:
        df['OPP_DEF_RTG_10'] = 115.0
    if 'OPP_PACE_10' not in df.columns:
        df['OPP_PACE_10'] = 100.0

    df['OPP_DEF_RTG_10'] = df['OPP_DEF_RTG_10'].fillna(115.0)
    df['OPP_PACE_10'] = df['OPP_PACE_10'].fillna(100.0)
    
    return df

def predict_and_rank(rankable: pd.DataFrame, model, features: list) -> pd.DataFrame:
    missing = [f for f in features if f not in rankable.columns]
    if missing:
        raise ValueError(f"Rankable rows are missing required features: {missing}")

    df = rankable.copy()
    df['PREDICTED_DELTA'] = model.predict(df[features])
    df['ML_PREDICTION'] = df['FP_ROLLING_10'] + df['PREDICTED_DELTA']
    df = df.sort_values('ML_PREDICTION', ascending=False).reset_index(drop=True)
    df['RANK'] = df.index + 1
    return df


def save_predictions(predictions: pd.DataFrame, game_date: pd.Timestamp) -> str:
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)

    output_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'GAME_DATE',
                   'SEASON_GAMES_PLAYED', 'FP_ROLLING_10', 'PREDICTED_DELTA', 'ML_PREDICTION', 'RANK']
    output_cols = [c for c in output_cols if c in predictions.columns]
    to_save = predictions[output_cols]

    if os.path.exists(PREDICTIONS_PATH):
        existing = pd.read_csv(PREDICTIONS_PATH, parse_dates=['GAME_DATE'])
        existing = existing[existing['GAME_DATE'] != game_date]  # idempotent re-run for the same date
        combined = pd.concat([existing, to_save], ignore_index=True)
    else:
        combined = to_save

    combined = combined.sort_values(['GAME_DATE', 'RANK']).reset_index(drop=True)
    combined.to_csv(PREDICTIONS_PATH, index=False)
    return PREDICTIONS_PATH


def run_daily_prediction(game_date: Optional[str] = None) -> pd.DataFrame:
    target_date = pd.Timestamp(game_date) if game_date else pd.Timestamp(date.today())
    date_str = target_date.strftime('%Y-%m-%d')
    season_start_year = current_season_start_year(target_date.date())
    historical_features_df = pd.read_csv(PROCESSED_PATH, parse_dates=['GAME_DATE'])
    print(f"Running daily prediction for {date_str}...")

    model, features = load_model_and_features()

    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"No processed feature store at {PROCESSED_PATH}. Run preprocess_live_data.py first.")
    processed = pd.read_csv(PROCESSED_PATH, parse_dates=['GAME_DATE'])

    schedule_path = os.path.join(SCHEDULE_DIRECTORY, f"schedule_{date_str}.csv")
    if not os.path.exists(schedule_path):
        raise FileNotFoundError(f"No schedule found at {schedule_path}. Run fetch_live_schedule.py for this date first.")
    schedule = pd.read_csv(schedule_path, parse_dates=['GAME_DATE'])

    injury_path = os.path.join(INJURY_DIRECTORY, f"injuries_{date_str}.csv")
    injuries = pd.read_csv(injury_path, parse_dates=['GAME_DATE']) if os.path.exists(injury_path) else None
    if injuries is None:
        print(f"  Warning: no injury report found at {injury_path}.")

    expected_active = build_expected_active_roster(processed, schedule, injuries, season_start_year, target_date)

    if expected_active.empty:
        print("  No teams/players found for this date -- nothing to predict.")
        return expected_active

    with_vacated = recompute_vacated_stats_for_tonight(expected_active)

    # Games-played floor decides who gets a final prediction -- applied
    # here, AFTER vacated stats, not before (see module docstring).
    before = len(with_vacated)
    rankable = with_vacated[with_vacated['SEASON_GAMES_PLAYED'] >= MIN_SEASON_GAMES].copy()
    print(f"  Dropped {before - len(rankable)} players with < {MIN_SEASON_GAMES} games this season "
          f"(rookies / just back from injury) from the rankable pool.")

    if rankable.empty:
        print("  No rankable players remain after filtering.")
        return rankable
    
    rankable = add_live_opponent_metrics(rankable, historical_features_df)

    predictions = predict_and_rank(rankable, model, features)
    output_path = save_predictions(predictions, target_date)

    print(f"Ranked {len(predictions)} players. Predictions saved to {output_path}")
    display_cols = [c for c in ['PLAYER_NAME', 'TEAM_ID', 'ML_PREDICTION', 'RANK'] if c in predictions.columns]
    print(predictions[display_cols].head(10).to_string(index=False))
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run today's (or a given date's) streamer predictions.")
    parser.add_argument("--date", type=str, default=None,
                         help="Date to predict for, 'YYYY-MM-DD'. Defaults to today.")
    args = parser.parse_args()
    run_daily_prediction(args.date)