#!/bin/bash
# cmd/evaluate_step.sh

# Usage: ./evaluate_step.sh <experiment_id> <iteration> <model_name> <adapter_type> <evaluator_type> <target_preference>

EXPERIMENT_ID=$1
ITERATION=$2
MODEL_NAME=$3
ADAPTER_TYPE=$4
EVALUATOR_TYPE=$5
TARGET_PREFERENCE=$6
EVOLUTION_METHOD=${7:-ga}
POPULATION_NAME=${8:-default}

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"
RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID/$POPULATION_NAME/$EVOLUTION_METHOD/$EVALUATOR_TYPE"

echo "Running Evaluation Step for $EXPERIMENT_ID ($EVOLUTION_METHOD) Iteration $ITERATION (Evaluator: $EVALUATOR_TYPE | Pop: $POPULATION_NAME)"

export PYTHONPATH=$PYTHONpath
python3 "$PROJECT_ROOT/src/evaluate.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --iteration "$ITERATION" \
    --model-name "$MODEL_NAME" \
    --adapter-type "$ADAPTER_TYPE" \
    --evaluator-type "$EVALUATOR_TYPE" \
    --target-preference "$TARGET_PREFERENCE" \
    --evolution-method "$EVOLUTION_METHOD" \
    --population-name "$POPULATION_NAME"
