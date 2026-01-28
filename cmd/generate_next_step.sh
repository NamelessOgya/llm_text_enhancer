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
POPULATION_NAME=${10:-default}

# Get absolute path to project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONpath="$PROJECT_ROOT/src"

# RESULT_DIR is optionally passed as 11th arg
RESULT_DIR=${11:-}

if [ -z "$RESULT_DIR" ]; then
    # Fallback to legacy construction if not provided
    RESULT_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID/$POPULATION_NAME/$EVOLUTION_METHOD/$EVALUATOR_TYPE"
fi

LOGIC_CONFIG=${12:-}

# Define Initial Population Dir
INITIAL_POP_DIR="$PROJECT_ROOT/result/$EXPERIMENT_ID/initial_population/$POPULATION_NAME"

echo "Running Generation Step for $EXPERIMENT_ID ($EVOLUTION_METHOD) Iteration $ITERATION (Pop: $POPULATION_NAME)"

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
    --ensemble-ratios "$ENSEMBLE_RATIOS" \
    --initial-population-dir "$INITIAL_POP_DIR" \
    --logic-config "$LOGIC_CONFIG"
