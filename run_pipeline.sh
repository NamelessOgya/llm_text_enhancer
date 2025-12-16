#!/bin/bash
# Generated pipeline script

set -e

# Experiment: pokemon_fire
echo "Starting Experiment: pokemon_fire"
if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_fire/iter0/metrics.json" ]; then
    echo "Running Iteration 0"
    ./cmd/generate_next_step.sh "pokemon_fire" "0" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_fire" "0" "gpt-4o" "Fire type pokemon"
else
    echo "Skipping Iteration 0 (already completed)"
fi

if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_fire/iter1/metrics.json" ]; then
    echo "Running Iteration 1"
    ./cmd/generate_next_step.sh "pokemon_fire" "1" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_fire" "1" "gpt-4o" "Fire type pokemon"
else
    echo "Skipping Iteration 1 (already completed)"
fi

if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_fire/iter2/metrics.json" ]; then
    echo "Running Iteration 2"
    ./cmd/generate_next_step.sh "pokemon_fire" "2" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_fire" "2" "gpt-4o" "Fire type pokemon"
else
    echo "Skipping Iteration 2 (already completed)"
fi

# Experiment: pokemon_water
echo "Starting Experiment: pokemon_water"
if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_water/iter0/metrics.json" ]; then
    echo "Running Iteration 0"
    ./cmd/generate_next_step.sh "pokemon_water" "0" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_water" "0" "gpt-4o" "Water type pokemon"
else
    echo "Skipping Iteration 0 (already completed)"
fi

if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_water/iter1/metrics.json" ]; then
    echo "Running Iteration 1"
    ./cmd/generate_next_step.sh "pokemon_water" "1" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_water" "1" "gpt-4o" "Water type pokemon"
else
    echo "Skipping Iteration 1 (already completed)"
fi

if [ ! -f "/Users/masashiueno/Desktop/text_enhancer/llm_text_enhancer/result/pokemon_water/iter2/metrics.json" ]; then
    echo "Running Iteration 2"
    ./cmd/generate_next_step.sh "pokemon_water" "2" "5" "gpt-4o" "Generate a description or name of a single real Pokemon species."
    ./cmd/evaluate_step.sh "pokemon_water" "2" "gpt-4o" "Water type pokemon"
else
    echo "Skipping Iteration 2 (already completed)"
fi

