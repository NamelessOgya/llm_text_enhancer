#!/bin/bash
set -e

EXPERIMENT_ID="test_eed_verify"
POP_SIZE=10
MODEL="gemini-2.0-flash-lite"
ADAPTER="gemini"
TASK="config/definitions/tasks/task_verify_eed.taml"
METHOD="eed"
EVALUATOR="dummy_eval"

# Cleanup
rm -rf "result/$EXPERIMENT_ID"

echo "=== Running Iteration 0 ==="
./cmd/generate_next_step.sh "$EXPERIMENT_ID" "0" "$POP_SIZE" "$MODEL" "$ADAPTER" "$TASK" "$METHOD" "" "$EVALUATOR"

echo "=== Mocking Metrics for Iteration 0 ==="
# We need to create result/test_eed_verify/eed/dummy_eval/row_0/iter0/metrics.json
# And assume text_0.txt etc exist.
ITER0_DIR="result/$EXPERIMENT_ID/$METHOD/$EVALUATOR/row_0/iter0"

# List files to make sure
ls "$ITER0_DIR/texts"

# Create dummy metrics using python for safety
python3 -c "
import json
pop_size = $POP_SIZE
metrics = []
for i in range(pop_size):
    score = min(0.99, 0.5 + i * 0.05)
    metrics.append({'file': f'text_{i}.txt', 'score': score, 'reason': 'Dummy reason'})
print(json.dumps(metrics))
" > "$ITER0_DIR/metrics.json"

echo "=== Running Iteration 1 (EED Logic) ==="
./cmd/generate_next_step.sh "$EXPERIMENT_ID" "1" "$POP_SIZE" "$MODEL" "$ADAPTER" "$TASK" "$METHOD" "" "$EVALUATOR"

echo "=== Verification Complete ==="
ls -l "result/$EXPERIMENT_ID/$METHOD/$EVALUATOR/row_0/iter1/texts"
cat "result/$EXPERIMENT_ID/$METHOD/$EVALUATOR/row_0/iter1/logic/creation_prompt_0.txt"
