#!/bin/bash
# Generated pipeline script

set -e

# Experiment: dummy_test
echo "Starting Experiment: dummy_test"
if [ ! -f "/app/result/dummy_test/iter0/metrics.json" ]; then
    echo "Running Iteration 0"
    ./cmd/generate_next_step.sh "dummy_test" "0" "3" "dummy" "Generate Pokemon"
    ./cmd/evaluate_step.sh "dummy_test" "0" "dummy" "Fire Type"
else
    echo "Skipping Iteration 0 (already completed)"
fi

if [ ! -f "/app/result/dummy_test/iter1/metrics.json" ]; then
    echo "Running Iteration 1"
    ./cmd/generate_next_step.sh "dummy_test" "1" "3" "dummy" "Generate Pokemon"
    ./cmd/evaluate_step.sh "dummy_test" "1" "dummy" "Fire Type"
else
    echo "Skipping Iteration 1 (already completed)"
fi

