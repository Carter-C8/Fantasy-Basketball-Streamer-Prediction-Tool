"""
aggregate_weekly.py

Phase 6 script: builds a FULL-WEEK forecast for the ISO week containing a
given date -- not just a running total of days already logged. This is
possible because the NBA schedule is released well in advance, so every
remaining day's opponent/home-away is already known.

Two kinds of rows make up a player's weekly total, distinguished by
IS_FORECAST:
  - Already-committed daily predictions (IS_FORECAST=False): whatever
    live_predict.py already logged for days up through the target date,
    using CALIBRATED_PREDICTION where recalibrate.py has run, each using
    THAT day's own actual injury report as it existed at the time.
  - Forecasted future days (IS_FORECAST=True): for every day later in
    the week, this script builds a real feature row -- each player's
    latest rolling stats, plus that day's ACTUAL known opponent/home-away
    -- and runs it through the model, same as live_predict.py would on
    the day itself.

IMPORTANT LIMITATION, by design, not oversight: forecasted days all use
TODAY's injury report, frozen and applied to every remaining day in the
week. The NBA doesn't publish injury reports more than a day or two out,
so there is no better information available at forecast time. This means:
  - If someone gets hurt (or clears Questionable) mid-week, every
    forecast made before that news breaks will be wrong for their
    remaining games until this script is re-run.
  - A player's rolling-stat features (FP_ROLLING_10, MIN_ROLLING_4, etc.)
    are also a single snapshot as of today, reused for every one of
    their forecasted games this week -- they don't update game-to-game
    within the forecast the way they would if each day were predicted
    fresh, since none of those future games' actual results exist yet.
  - Re-running this script daily is what keeps it accurate: each day,
    more of the week moves from "forecasted" into "already committed"
    (with that day's real injury report), narrowing the window of
    frozen-assumption forecasting to whatever's still ahead.

The latest known bias correction (from recalibrate.py's calibration log)
is applied to forecasted rows too, so they're calibrated consistently
with already-committed days rather than left as raw model output.

Reuses functions directly from fetch_live_schedule.py and live_predict.py
rather than reimplementing feature assembly and prediction -- if either
of those files' functions change shape, this import needs updating too.

Usage:
    python aggregate_weekly.py
    python aggregate_weekly.py --date 2026-01-15
"""

import argparse
import os
import sys
from datetime import date
from typing import Optional

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_STREAMER_ML_MODEL_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(NBA_STREAMER_ML_MODEL_DIR)

DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data_pipeline", "nba_live_data")
PROCESSED_PATH = os.path.join(DATA_DIRECTORY, "processed", "player_features.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIRECTORY, "predictions", "daily_predictions.csv")
SCHEDULE_DIRECTORY = os.path.join(DATA_DIRECTORY, "schedule")
INJURY_DIRECTORY = os.path.join(DATA_DIRECTORY, "injuries")
CALIBRATION_LOG_PATH = os.path.join(DATA_DIRECTORY, "predictions", "calibration_log.csv")
WEEKLY_TOTALS_PATH = os.path.join(DATA_DIRECTORY, "predictions", "weekly_totals.csv")
WEEKLY_INDIVIDUAL_GAMES_PATH = os.path.join(DATA_DIRECTORY, "predictions", "weekly_individual_games.csv")

DATA_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "data_pipeline")
sys.path.insert(0, DATA_PIPELINE_DIR)


from fetch_live_schedule import fetch_schedule_for_date  # noqa: E402

sys.path.insert(0, SCRIPT_DIR)
from live_predict import (  # noqa: E402
    load_model_and_features, current_season_start_year, build_expected_active_roster,
    recompute_vacated_stats_for_tonight, predict_and_rank, MIN_SEASON_GAMES,
)


def get_iso_week_dates(target_date: pd.Timestamp) -> list:
    """All 7 calendar dates in target_date's ISO week (Monday-Sunday)."""
    iso = target_date.isocalendar()
    # Use DateOffset and ensure the weekday is a standard Python int
    monday = target_date - pd.DateOffset(days=int(iso.weekday) - 1)
    return [monday + pd.DateOffset(days=i) for i in range(7)]


def load_or_fetch_schedule(day: pd.Timestamp) -> pd.DataFrame:
    date_str = day.strftime('%Y-%m-%d')
    schedule_path = os.path.join(SCHEDULE_DIRECTORY, f"schedule_{date_str}.csv")
    if os.path.exists(schedule_path):
        return pd.read_csv(schedule_path, parse_dates=['GAME_DATE'])

    print(f"  No cached schedule for {date_str} -- fetching it now.")
    schedule = fetch_schedule_for_date(date_str)
    os.makedirs(SCHEDULE_DIRECTORY, exist_ok=True)
    schedule.to_csv(schedule_path, index=False)
    return schedule


def load_predictions_for_week(week_dates: list) -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"No predictions log at {PREDICTIONS_PATH}. Run live_predict.py first.")

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=['GAME_DATE'])
    week_predictions = predictions[predictions['GAME_DATE'].isin(week_dates)].copy()

    # Use the calibrated prediction where available (recalibrate.py has
    # run for that day); fall back to the raw prediction otherwise.
    if 'CALIBRATED_PREDICTION' in week_predictions.columns:
        week_predictions['FINAL_PREDICTION'] = week_predictions['CALIBRATED_PREDICTION'].fillna(
            week_predictions['ML_PREDICTION']
        )
    else:
        week_predictions['FINAL_PREDICTION'] = week_predictions['ML_PREDICTION']

    return week_predictions


def get_latest_bias_correction(as_of_date: pd.Timestamp) -> Optional[float]:
    """Most recent bias correction on or before as_of_date. If
    recalibrate.py has already run for as_of_date (the normal pipeline
    order: ... -> live_predict -> recalibrate -> aggregate_weekly), this
    picks up TODAY's freshly computed correction -- otherwise it falls
    back to the most recent prior day's."""
    if not os.path.exists(CALIBRATION_LOG_PATH):
        return None
    log = pd.read_csv(CALIBRATION_LOG_PATH, parse_dates=['GAME_DATE'])
    prior_or_today = log[log['GAME_DATE'] <= as_of_date]
    if prior_or_today.empty:
        return None
    return prior_or_today.sort_values('GAME_DATE').iloc[-1]['BIAS_CORRECTION']


def generate_future_forecasts(week_dates: list, target_date: pd.Timestamp,
                               processed: pd.DataFrame, model, features: list,
                               today_injuries: Optional[pd.DataFrame],
                               season_start_year: int) -> pd.DataFrame:
    """Builds a real prediction for every remaining scheduled day this
    week (dates > target_date), reusing live_predict.py's own feature
    assembly and prediction functions. today_injuries is held constant
    across every one of these days -- see the module docstring for why."""
    future_dates = [d for d in week_dates if d > target_date]
    if not future_dates:
        return pd.DataFrame()

    # Computed once and reused across every future date below, rather
    # than recomputed inside build_expected_active_roster() on each call
    # -- these are load_latest_player_state()/count_season_games()'s own
    # outputs, valid as long as `processed` doesn't change mid-loop here.
    from live_predict import load_latest_player_state, count_season_games
    precomputed_latest_state = load_latest_player_state(processed)
    precomputed_season_games = count_season_games(processed, season_start_year)

    forecasts = []
    for day in future_dates:
        schedule = load_or_fetch_schedule(day)
        if schedule.empty:
            continue

        expected_active = build_expected_active_roster(
            processed, schedule, today_injuries, season_start_year, day,
            latest_state=precomputed_latest_state, season_games=precomputed_season_games,
        )
        if expected_active.empty:
            continue

        with_vacated = recompute_vacated_stats_for_tonight(expected_active)
        rankable = with_vacated[with_vacated['SEASON_GAMES_PLAYED'] >= MIN_SEASON_GAMES].copy()
        if rankable.empty:
            continue

        day_forecast = predict_and_rank(rankable, model, features)
        forecasts.append(day_forecast)

    if not forecasts:
        return pd.DataFrame()
    return pd.concat(forecasts, ignore_index=True)

def aggregate_week(target_date: pd.Timestamp) -> pd.DataFrame:
    week_dates = get_iso_week_dates(target_date)
    print(f"Forecasting week of {week_dates[0].date()} - {week_dates[-1].date()} "
          f"(as of {target_date.date()})...")

    # --- Already-committed days (today and earlier) ---
    week_predictions = load_predictions_for_week(week_dates)
    week_predictions = week_predictions.copy()
    week_predictions['IS_FORECAST'] = False

    # --- Forecasted days (later in the week) ---
    model, features = load_model_and_features()
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"No processed feature store at {PROCESSED_PATH}. Run preprocess_live_data.py first.")
    processed = pd.read_csv(PROCESSED_PATH, parse_dates=['GAME_DATE'])
    season_start_year = current_season_start_year(target_date.date())

    injury_path = os.path.join(INJURY_DIRECTORY, f"injuries_{target_date.strftime('%Y-%m-%d')}.csv")
    today_injuries = pd.read_csv(injury_path, parse_dates=['GAME_DATE']) if os.path.exists(injury_path) else None
    if today_injuries is None:
        print(f"  Warning: no injury report for {target_date.date()} -- future-day forecasts "
              f"won't reflect any current absences.")

    future_forecasts = generate_future_forecasts(
        week_dates, target_date, processed, model, features, today_injuries, season_start_year
    )

    if not future_forecasts.empty:
        future_forecasts = future_forecasts.copy()
        future_forecasts['IS_FORECAST'] = True
        latest_bias = get_latest_bias_correction(target_date)
        if latest_bias is not None:
            future_forecasts['FINAL_PREDICTION'] = future_forecasts['ML_PREDICTION'] + latest_bias
            print(f"  Applied latest known bias correction ({latest_bias:+.2f}) to "
                  f"{len(future_forecasts)} forecasted player-games.")
        else:
            future_forecasts['FINAL_PREDICTION'] = future_forecasts['ML_PREDICTION']
            print("  No bias correction available yet (recalibrate.py hasn't accumulated "
                  "enough history) -- forecasted rows are uncalibrated.")

    all_week = pd.concat([week_predictions, future_forecasts], ignore_index=True) \
        if not future_forecasts.empty else week_predictions

    if all_week.empty:
        print("  No predictions or forecasts available for this week.")
        return all_week, pd.DataFrame()

    season_start = pd.Timestamp('2026-10-20') # Needs to be updated every season
    target_dt = pd.to_datetime(target_date)
    
    # 2. Force any game happening today or in the future to be a forecast
    all_week['IS_FORECAST'] = all_week['GAME_DATE'] >= target_dt
    
    weekly = all_week.groupby('PLAYER_ID').agg(
        PLAYER_NAME=('PLAYER_NAME', 'first'),
        TEAM_ID=('TEAM_ID', 'last'),
        GAMES_LOGGED=('IS_FORECAST', lambda s: int((~s[all_week.loc[s.index, 'GAME_DATE'] >= season_start]).sum())),
        GAMES_FORECASTED=('IS_FORECAST', lambda s: int(s.sum())),
        PREDICTED_WEEKLY_FP=('FINAL_PREDICTION', 'sum'),
    ).reset_index()
    weekly['TOTAL_GAMES_THIS_WEEK'] = weekly['GAMES_LOGGED'] + weekly['GAMES_FORECASTED']

    weekly['ISO_YEAR'] = target_date.isocalendar().year
    weekly['ISO_WEEK'] = target_date.isocalendar().week
    weekly = weekly.sort_values('PREDICTED_WEEKLY_FP', ascending=False).reset_index(drop=True)
    weekly['RANK'] = weekly.index + 1

    return weekly, all_week


def save_weekly_totals(weekly: pd.DataFrame) -> str:
    os.makedirs(os.path.dirname(WEEKLY_TOTALS_PATH), exist_ok=True)

    if weekly.empty:
        return WEEKLY_TOTALS_PATH

    iso_year, iso_week = weekly['ISO_YEAR'].iloc[0], weekly['ISO_WEEK'].iloc[0]

    if os.path.exists(WEEKLY_TOTALS_PATH):
        existing = pd.read_csv(WEEKLY_TOTALS_PATH)
        existing = existing[~((existing['ISO_YEAR'] == iso_year) & (existing['ISO_WEEK'] == iso_week))]
        combined = pd.concat([existing, weekly], ignore_index=True)
    else:
        combined = weekly

    combined = combined.sort_values(['ISO_YEAR', 'ISO_WEEK', 'RANK']).reset_index(drop=True)
    combined.to_csv(WEEKLY_TOTALS_PATH, index=False)
    return WEEKLY_TOTALS_PATH

def save_weekly_individual_games(all_week: pd.DataFrame):
    iso_year, iso_week = all_week['GAME_DATE'].iloc[0].isocalendar().year, all_week['GAME_DATE'].iloc[0].isocalendar().week
    all_week['ISO_YEAR'] = iso_year
    all_week['ISO_WEEK'] = iso_week

    if os.path.exists(WEEKLY_INDIVIDUAL_GAMES_PATH):
        existing = pd.read_csv(WEEKLY_INDIVIDUAL_GAMES_PATH, parse_dates=['GAME_DATE'])
        # Drop the existing data for this specific week before appending the fresh run
        existing = existing[~((existing['ISO_YEAR'] == iso_year) & (existing['ISO_WEEK'] == iso_week))]
        combined = pd.concat([existing, all_week], ignore_index=True)
    else:
        combined = all_week

    combined.to_csv(WEEKLY_INDIVIDUAL_GAMES_PATH, index=False)

def run_weekly_aggregation(game_date: Optional[str] = None):
    base_date = pd.Timestamp(game_date) if game_date else pd.Timestamp(date.today())
    
    # Run for Current Week (offset 0) and Next Week (offset 1)
    for offset in [0, 1]:
        target_date = base_date + pd.DateOffset(days=7 * offset)
        weekly, all_week_games = aggregate_week(target_date) # We will update aggregate_week next
        
        if not weekly.empty:
            save_weekly_totals(weekly)
            save_weekly_individual_games(all_week_games)
            print(f"✅ Processed Week {target_date.isocalendar().week} (Offset {offset})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate this week's daily predictions into weekly totals.")
    parser.add_argument("--date", type=str, default=None,
                         help="Date whose ISO week to aggregate, 'YYYY-MM-DD'. Defaults to today.")
    args = parser.parse_args()
    run_weekly_aggregation(args.date)