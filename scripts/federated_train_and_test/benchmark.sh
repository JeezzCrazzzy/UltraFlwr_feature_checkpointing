#!/bin/bash

# navigate to directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH

cd ../../

# Read CLIENT_CONFIG from Python file
CLIENT_CONFIG_FILE="./FedYOLO/config.py"
if [[ ! -f "$CLIENT_CONFIG_FILE" ]]; then
    echo "Error: $CLIENT_CONFIG_FILE not found"
    exit 1
fi

# Install FedYOLO from setup.py, uncomment if already installed
if [[ -f "setup.py" ]]; then
    echo "Installing FedYOLO package..."
    pip install --no-cache-dir -e .
else
    echo "Error: setup.py not found. Cannot install FedYOLO."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# List of datasets and strategies to test
DATASET_NAME_LIST=("baseline" "m2cai16" "mydataset")
STRATEGY_LIST=("FedAvg" "FedHeadAvg" "FedHeadMedian" "FedNeckAvg" "FedNeckMedian" "FedBackboneAvg" "FedBackboneMedian" "FedNeckHeadAvg" "FedNeckHeadMedian")

# Function to check if strategy contains head, neck, or backbone
should_skip_server() {
    local strategy=$1
    if [[ "$strategy" == *"Head"* || "$strategy" == *"Neck"* || "$strategy" == *"Backbone"* ]]; then
        return 0  # true (skip server-server or server-client tests)
    else
        return 1  # false (run all tests)
    fi
}

# Loop over datasets and strategies
for DATASET_NAME in "${DATASET_NAME_LIST[@]}"; do
    for STRATEGY in "${STRATEGY_LIST[@]}"; do
        echo "===================================================================="
        echo "Running benchmark for DATASET_NAME=${DATASET_NAME}, STRATEGY=${STRATEGY}"
        echo "===================================================================="

        # Update config.py with current dataset and strategy
        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" $CLIENT_CONFIG_FILE
        sed -i "s/^\s*'strategy': .*/    'strategy': '${STRATEGY}',/" $CLIENT_CONFIG_FILE

        # Run data partitioning (uncomment if needed)
        # python FedYOLO/data_partitioner/fed_split.py >> logs/data_partition_log.txt 2>&1

        # Run federated training
        echo "Starting federated training..."
        bash scripts/federated_train_and_test/run.sh >> "logs/benchmark_${DATASET_NAME}_${STRATEGY}.txt" 2>&1

        # Wait a bit before running tests
        sleep 10

        # Run tests
        echo "Running tests..."
        bash FedYOLO/test/test.sh >> "logs/test_${DATASET_NAME}_${STRATEGY}.txt" 2>&1

        echo "Completed benchmark for ${DATASET_NAME} with ${STRATEGY}"
        echo "----------------------------------------"
    done
done

echo "All benchmarks completed."
