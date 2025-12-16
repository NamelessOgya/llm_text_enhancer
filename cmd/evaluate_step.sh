#!/bin/bash
# cmd/evaluate_step.sh

# Usage: ./evaluate_step.sh <experiment_id> <iteration> <model_name> <target_preference>

EXPERIMENT_ID=$1
ITERATION=$2
MODEL_NAME=$3
TARGET_PREFERENCE=$4

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"
RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID"

echo "Running Evaluation Step for $EXPERIMENT_ID Iteration $ITERATION"

export PYTHONPATH=$PYTHONpath
python3 "$PROJECT_ROOT/src/evaluate.py" \
    --iteration "$ITERATION" \
    --model-name "$MODEL_NAME" \
    --target-preference "$TARGET_PREFERENCE" \
    --result-dir "$RESULT_DIR"
