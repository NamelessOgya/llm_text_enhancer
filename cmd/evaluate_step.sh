#!/bin/bash
# cmd/evaluate_step.sh

# Usage: ./evaluate_step.sh <experiment_id> <iteration> <model_name> <adapter_type> <evaluator_type> <target_preference>

EXPERIMENT_ID=$1
ITERATION=$2
MODEL_NAME=$3
ADAPTER_TYPE=$4
EVALUATOR_TYPE=$5
TARGET_PREFERENCE=$6

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"
RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID"

echo "Running Evaluation Step for $EXPERIMENT_ID Iteration $ITERATION (Evaluator: $EVALUATOR_TYPE)"

export PYTHONPATH=$PYTHONpath
python3 "$PROJECT_ROOT/src/evaluate.py" \
    --iteration "$ITERATION" \
    --model-name "$MODEL_NAME" \
    --adapter-type "$ADAPTER_TYPE" \
    --evaluator-type "$EVALUATOR_TYPE" \
    --target-preference "$TARGET_PREFERENCE" \
    --result-dir "$RESULT_DIR"
