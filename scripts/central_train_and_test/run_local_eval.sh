#!/bin/bash

# This script was used to test one model on each data partition of the dataset.
# as well as the entire dataset.
# This script is used to generate the results in Table 3 of the paper.

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
LOG_FILE="logs/local_eval_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"

# Get dataset path from config
DATASET_PATH=$(python -c "from FedYOLO.config import DATASET_PATH; print(DATASET_PATH)")

# Define global model path (you may need to adjust this based on your setup)
GLOBAL_MODEL_PATH="weights/global_model.pt"

echo "Starting local evaluation..."
echo "Dataset: $DATASET_NAME"
echo "Strategy: $STRATEGY_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Global model path: $GLOBAL_MODEL_PATH"
echo "Log file: $LOG_FILE"

# Run the local evaluation script
python scripts/central_train_and_test/local_test_only.py --data "$DATASET_PATH" --model "$GLOBAL_MODEL_PATH" | tee "$LOG_FILE"

echo "Local evaluation completed."