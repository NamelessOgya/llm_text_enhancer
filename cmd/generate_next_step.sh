#!/bin/bash
# cmd/generate_next_step.sh

# Usage: ./generate_next_step.sh <experiment_id> <iteration> <population_size> <model_name> <task_definition>

EXPERIMENT_ID=$1
ITERATION=$2
POPULATION_SIZE=$3
MODEL_NAME=$4
TASK_DEFINITION=$5

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"

# Define result dir relative to project root
RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID"

echo "Running Generation Step for $EXPERIMENT_ID Iteration $ITERATION"

export PYTHONPATH=$PYTHONpath
python3 "$PROJECT_ROOT/src/generate.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --iteration "$ITERATION" \
    --population-size "$POPULATION_SIZE" \
    --model-name "$MODEL_NAME" \
    --task-definition "$TASK_DEFINITION" \
    --result-dir "$RESULT_DIR"
