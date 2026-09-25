#!/usr/bin/env bash

# ==============================================================================
# Daily Automated NBA Fantasy Streamer Pipeline
# ==============================================================================
# - Exits immediately if any step encounters an error (set -e)
# - Logs progress with timestamps
# - Resolves absolute paths dynamically to support cron jobs and CI/CD
# ==============================================================================

set -eo pipefail

# 1. Resolve Project Root Directory (adjust if script lives in a subdirectory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Optional: Output log file with date stamp
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +'%Y-%m-%d').log"

# Route stdout and stderr to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "================ Starting Daily Pipeline ================"

# 2. Activate Python Virtual Environment
if [ -d "$PROJECT_ROOT/venv" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
  log "Activated virtual environment at $PROJECT_ROOT/venv"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
  log "Activated virtual environment at $PROJECT_ROOT/.venv"
else
  log "Warning: No virtual environment found. Using system Python: $(which python3)"
fi

# 3. Step 1: Fetch New Box Scores & Update Live Data
log "Step 1/6: Fetching latest box scores and refreshing player positions cache..."
python3 data_pipeline/retrieve_live_data.py 

# 4. Step 2: Run Preprocessing Pipeline
log "Step 2/6: Preprocessing completed games and computing rolling feature store..."
python3 data_pipeline/preprocess_live_data.py #[cite: 3]

# 5. Step 3: Fetch Today's Game Schedule
log "Step 3/6: Fetching tonight's NBA game schedule..."
python3 data_pipeline/fetch_live_schedule.py #[cite: 2]

# 6. Step 4: Fetch NBA Official Injury Report
log "Step 4/6: Pulling official NBA injury report snapshot..."
python3 data_pipeline/fetch_live_injury_report.py #[cite: 1]

# 7. Step 5: Run Daily Prediction
log "Step 5/6: Generating daily streamer predictions..."
python3 NBA_STREAMER_ML_MODEL/live_inference_scripts/live_predict.py

# 8. Step 6: Aggregate Weekly Projections & Future Game Forecasts
log "Step 6/6: Aggregating multi-week projections and updating individual games..."
python3 NBA_STREAMER_ML_MODEL/live_inference_scripts/aggregate_weekly.py

log "================ Pipeline Completed Successfully ================"