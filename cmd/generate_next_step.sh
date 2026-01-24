#!/bin/bash
# cmd/generate_next_step.sh

# Usage: ./generate_next_step.sh <experiment_id> <iteration> <population_size> <model_name> <adapter_type> <task_definition>

EXPERIMENT_ID=$1
ITERATION=$2
POPULATION_SIZE=$3
MODEL_NAME=$4
ADAPTER_TYPE=$5
TASK_DEFINITION=$6
EVOLUTION_METHOD=${7:-ga} # Default to ga
ENSEMBLE_RATIOS=${8:-""}
EVALUATOR_TYPE=${9:-llm}

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"

# Define result dir relative to project root (Split by Logic AND Evaluator)
RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID/$EVOLUTION_METHOD/$EVALUATOR_TYPE"

echo "Running Generation Step for $EXPERIMENT_ID ($EVOLUTION_METHOD) Iteration $ITERATION"

export PYTHONPATH=$PYTHONpath
python3 "$PROJECT_ROOT/src/generate.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --iteration "$ITERATION" \
    --population-size "$POPULATION_SIZE" \
    --model-name "$MODEL_NAME" \
    --adapter-type "$ADAPTER_TYPE" \
    --task-definition "$TASK_DEFINITION" \
    --result-dir "$RESULT_DIR" \
    --evolution-method "$EVOLUTION_METHOD" \
    --ensemble-ratios "$ENSEMBLE_RATIOS"
