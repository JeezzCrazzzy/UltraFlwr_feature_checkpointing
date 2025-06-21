#!/bin/bash

# This script runs local training and testing on the specified dataset partitions.
# Models are trained and tested on the same dataset partition.
# The script is used to fill the Central Train in Table 2.

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Install FedYOLO from setup.py
if [[ -f "setup.py" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e .
else
    echo "Error: setup.py not found. Cannot install FedYOLO."
    exit 1
fi

# Get dataset name and strategy from config
DATASET_NAME=$(python -c "from FedYOLO.config import SPLITS_CONFIG; print(SPLITS_CONFIG['dataset_name'])")
STRATEGY_NAME=$(python -c "from FedYOLO.config import SERVER_CONFIG; print(SERVER_CONFIG['strategy'])")

# Create logs directory if it doesn't exist
mkdir -p logs

# Define log file path
LOG_FILE="logs/local_train_and_test_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"

# Get dataset path from config
DATASET_PATH=$(python -c "from FedYOLO.config import DATASET_PATH; print(DATASET_PATH)")

echo "Starting local training and testing..."
echo "Dataset: $DATASET_NAME"
echo "Strategy: $STRATEGY_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Log file: $LOG_FILE"

# Run the local training and testing script
python scripts/central_train_and_test/local_train_and_test.py --data "$DATASET_PATH" | tee "$LOG_FILE"

echo "Local training and testing completed."