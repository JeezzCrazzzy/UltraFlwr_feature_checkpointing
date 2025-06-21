#!/bin/bash

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

# Define script paths
SERVER_SCRIPT="FedYOLO/train/yolo_server.py"
CLIENT_SCRIPT="FedYOLO/train/yolo_client.py"

# Define log file paths
SERVER_LOG="logs/server_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
CLIENT_LOG_PREFIX="logs/client"

# Function to check if server is ready
check_server_ready() {
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for server to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if netstat -an | grep -q ":8080.*LISTEN"; then
            echo "Server is ready on port 8080"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "Error: Server failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Start the server
echo "Starting server..."
python "$SERVER_SCRIPT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
if ! check_server_ready; then
    echo "Failed to start server. Check server logs: $SERVER_LOG"
    exit 1
fi

# Get number of clients from config
NUM_CLIENTS=$(python -c "from FedYOLO.config import NUM_CLIENTS; print(NUM_CLIENTS)")

# Start clients
for CLIENT_CID in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $CLIENT_CID..."
    CLIENT_DATA_PATH=$(python -c "from FedYOLO.config import CLIENT_CONFIG; print(CLIENT_CONFIG[$CLIENT_CID]['data_path'])")
    CLIENT_LOG="${CLIENT_LOG_PREFIX}_${CLIENT_CID}_log_${DATASET_NAME}_${STRATEGY_NAME}.txt"
    python "$CLIENT_SCRIPT" --cid="$CLIENT_CID" --data_path="$CLIENT_DATA_PATH" > "$CLIENT_LOG" 2>&1 &
done

# Wait for all processes to complete
wait

echo "All processes completed."
