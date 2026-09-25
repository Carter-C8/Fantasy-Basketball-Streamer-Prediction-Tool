"""
preprocess.py

Phase 1 data script: turns raw box scores (from fetch_boxscores.py) into
the rolling-average / vacated-stat features the model expects.

This is the LIVE-SERVING version of preprocessing, not the training
version -- it deliberately excludes label_streamers / filter_out_top_players
/ partition_datasets / save_processed_data from preprocess_nba_data.ipynb,
since "who counts as a streamer" is a training-set-construction concept
with no role in generating a feature row for prediction. Free-agent
eligibility at serving time gets checked live against ESPN instead
(Phase 8), not baked in here.

This script processes COMPLETED games only -- it builds up the rolling
feature history used as lookback for predictions. It does NOT construct
tonight's not-yet-played feature row (opponent, home/away, vacated
minutes for a game that hasn't happened) -- that's predict_today.py's
job, using live schedule + injury-report data (Phase 2), since none of
that exists in a box score before the game is played.

Not implemented here: any OPP_DEF_RTG_10 / OPP_PACE_10-style opponent
features added later outside this thread -- port that logic in separately
if you're using it.

Designed to be re-run daily WITHOUT reprocessing the entire history:
  - Finds raw rows newer than the latest date already in the processed
    store.
  - Pulls the FULL history (not just the new rows) for every player who
    appears in that new date range -- rolling averages need that lookback
    to be computed correctly.
  - Recomputes the feature pipeline over just that subset of players
    (typically a few hundred, not the whole multi-season roster).
  - Appends only the newly computed rows to the processed store. Rows
    already processed are never touched or recomputed: each row's rolling
    features depend only on that player's STRICTLY PRIOR games (via
    shift(1)), so a new game today can never retroactively change
    yesterday's already-computed row.

Usage:
    python preprocess.py
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "nba_live_data")
RAW_PATH = os.path.join(DATA_DIRECTORY, "scores", "player_boxscores.csv")
PROCESSED_PATH = os.path.join(DATA_DIRECTORY, "processed", "player_features.csv")


# --- Feature-engineering functions, carried over from preprocess_nba_data.ipynb ---

def clean_data(raw_data):
    df = raw_data.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Fill missing stat values with 0. Extended vs. the original notebook
    # to also cover FG3M/FGM/FGA/FTM/FTA, which convert_fantasy_points
    # consumes but the notebook's version never NA-filled.
    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN',
                 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA']
    for col in stat_cols:
        if col in df.columns:
            if col == 'MIN' and df[col].dtype == 'object':
                df[col] = df[col].fillna('0:00')
                df[col] = df[col].apply(
                    lambda x: float(str(x).split(':')[0]) + (float(str(x).split(':')[1]) / 60)
                    if ':' in str(x) else float(x)
                )
            else:
                df[col] = df[col].fillna(0).astype(float)

    df['HOME_GAME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
    df['OPPONENT'] = df['MATCHUP'].str.split(' ').str[-1]
    return df


def convert_fantasy_points(clean_df):
    df = clean_df.copy()
    df['FANTASY_PTS'] = (
        df['PTS'] * 1.0 +
        df['REB'] * 1.0 +
        df['AST'] * 2 +
        df['STL'] * 4.0 +
        df['BLK'] * 4.0 +
        df['FG3M'] * 1.0 +
        df['FGM'] * 2.0 -
        df['FGA'] * 1.0 +
        df['FTM'] * 1.0 -
        df['FTA'] * 1.0 -
        df['TOV'] * 2.0
    )
    return df


def create_rolling_avgs(fantasy_points_df):
    df = fantasy_points_df.sort_values(by=['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    grouped_player = df.groupby('PLAYER_ID')

    df['FP_ROLLING_3'] = grouped_player['FANTASY_PTS'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df['MIN_ROLLING_3'] = grouped_player['MIN'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)

    df['FP_ROLLING_4'] = grouped_player['FANTASY_PTS'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    df['MIN_ROLLING_4'] = grouped_player['MIN'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)

    df['FP_ROLLING_10'] = grouped_player['FANTASY_PTS'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
    df['MIN_ROLLING_10'] = grouped_player['MIN'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)

    df['FP_PER_MIN_10'] = np.where(
        df['MIN_ROLLING_10'] > 0,
        df['FP_ROLLING_10'] / df['MIN_ROLLING_10'], 0
    )

    normal_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
    for stat in normal_stats:
        df[f'{stat}_ROLLING_4'] = grouped_player[stat].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    return df


def add_vacated_stats(rolling_avgs_df):
    df = rolling_avgs_df.copy()
    df['EXPECTED_MIN'] = df['MIN_ROLLING_4']
    df['EXPECTED_PTS'] = df['PTS_ROLLING_4']
    df['EXPECTED_REB'] = df['REB_ROLLING_4']
    df['EXPECTED_AST'] = df['AST_ROLLING_4']

    team_game_expected = df.groupby(['TEAM_ID', 'GAME_DATE'], observed=False)[
        ['EXPECTED_MIN', 'EXPECTED_PTS', 'EXPECTED_REB', 'EXPECTED_AST']].sum().reset_index()

    team_game_expected.rename(columns={
        'EXPECTED_MIN': 'TEAM_ACTIVE_MIN',
        'EXPECTED_PTS': 'TEAM_ACTIVE_PTS',
        'EXPECTED_REB': 'TEAM_ACTIVE_REB',
        'EXPECTED_AST': 'TEAM_ACTIVE_AST'
    }, inplace=True)

    df = pd.merge(df, team_game_expected, on=['TEAM_ID', 'GAME_DATE'], how='left')

    df['TEAM_VACATED_MINUTES'] = (240.0 - df['TEAM_ACTIVE_MIN']).clip(lower=0)
    df['TEAM_VACATED_PTS'] = (115.0 - df['TEAM_ACTIVE_PTS']).clip(lower=0)
    df['TEAM_VACATED_REB'] = (45.0 - df['TEAM_ACTIVE_REB']).clip(lower=0)
    df['TEAM_VACATED_AST'] = (25.0 - df['TEAM_ACTIVE_AST']).clip(lower=0)

    df = df.drop(columns=[
        'EXPECTED_MIN', 'EXPECTED_PTS', 'EXPECTED_REB', 'EXPECTED_AST',
        'TEAM_ACTIVE_MIN', 'TEAM_ACTIVE_PTS', 'TEAM_ACTIVE_REB',
        'TEAM_ACTIVE_AST'])
    return df

def add_opponent_metrics(df):
    df = df.copy()
    
    # 1. Aggregate individual player stats to their Team for every game
    team_game_stats = df.groupby(['GAME_DATE', 'TEAM_ID', 'OPPONENT'], observed=False)[
            ['PTS', 'FGA', 'FTA', 'TOV']].sum().reset_index()

    # 2. Approximate Possessions using FGA, FTA, and TOV
    team_game_stats['POSSESSIONS'] = team_game_stats['FGA'] + (0.44 * team_game_stats['FTA']) + team_game_stats['TOV']

    # 3. Define defense vs. offense relative to a team
    defense_df = team_game_stats[['GAME_DATE', 'OPPONENT', 'PTS', 'POSSESSIONS']].copy()
    defense_df.rename(columns={
            'OPPONENT': 'TEAM_ID',
            'PTS': 'PTS_ALLOWED',
            'POSSESSIONS': 'PACE_ALLOWED'
    }, inplace=True)

    # 4. Calculate Rolling Averages for the Defense
    defense_df = defense_df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    grouped_def = defense_df.groupby('TEAM_ID')

    # Opponent Defensive Generosity (Points allowed per game)
    # .shift(1) prevents look-ahead bias/data leakage
    defense_df['OPP_DEF_RTG_10'] = grouped_def['PTS_ALLOWED'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(115.0)

    # Opponent Pace (Possessions allowed per game)
    defense_df['OPP_PACE_10'] = grouped_def['PACE_ALLOWED'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(100.0)

    # 5. Merge the defensive context back into the main player dataframe
    df = df.merge(
            defense_df[['GAME_DATE', 'TEAM_ID', 'OPP_DEF_RTG_10', 'OPP_PACE_10']],
            left_on=['GAME_DATE', 'OPPONENT'],
            right_on=['GAME_DATE', 'TEAM_ID'],
            how='left',
            suffixes=('', '_DEF')
    )

    # Clean up the duplicate TEAM_ID column from the merge
    df = df.drop(columns=['TEAM_ID_DEF'])

    return df


def run_feature_pipeline(raw_slice: pd.DataFrame) -> pd.DataFrame:
    """Runs the full clean -> fantasy points -> rolling -> vacated pipeline
    on whatever slice of raw rows it's given."""
    df = clean_data(raw_slice)
    df = convert_fantasy_points(df)
    df = create_rolling_avgs(df)
    df = add_vacated_stats(df)
    df = add_opponent_metrics(df)
    return df


def update_features() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"No raw box scores found at {RAW_PATH}. Run fetch_boxscores.py first.")

    raw = pd.read_csv(RAW_PATH, parse_dates=['GAME_DATE'])
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    existing = None
    last_processed_date = None
    if os.path.exists(PROCESSED_PATH):
        existing = pd.read_csv(PROCESSED_PATH, parse_dates=['GAME_DATE'])
        last_processed_date = existing['GAME_DATE'].max()

    if last_processed_date is None:
        print("No existing processed store found -- processing full history.")
        new_raw = raw
    else:
        new_raw = raw[raw['GAME_DATE'] > last_processed_date]
        print(f"Processed store found (through {last_processed_date.date()}). "
              f"{len(new_raw)} new raw rows to process.")

    if len(new_raw) == 0:
        print("Nothing new to process.")
        return existing

    # Rolling features need each affected player's full history as
    # lookback, not just their new rows -- pull their complete history,
    # recompute the pipeline over that, then keep only the new dates.
    affected_players = new_raw['PLAYER_ID'].unique()
    working_set = raw[raw['PLAYER_ID'].isin(affected_players)]

    computed = run_feature_pipeline(working_set)
    cutoff = last_processed_date if last_processed_date is not None else pd.Timestamp.min
    new_features = computed[computed['GAME_DATE'] > cutoff]

    combined = pd.concat([existing, new_features], ignore_index=True) if existing is not None else new_features
    combined = combined.sort_values(['GAME_DATE', 'TEAM_ID', 'PLAYER_ID']).reset_index(drop=True)
    combined.to_csv(PROCESSED_PATH, index=False)

    print(f"Added {len(new_features)} newly processed rows. "
          f"Processed store now has {len(combined)} total rows, saved to {PROCESSED_PATH}")
    return combined


if __name__ == "__main__":
    update_features()