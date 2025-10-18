#!/bin/bash

# This script runs the benchmark client with different batch sizes and repetitions.
# Before running this script, ensure that the server is up and running.
# Check the config/default_config.yaml for server URL and other configurations.
# Activate the virtual environment if needed

model_name="qwen2.5-vl-7b"
engine="vllm"
output_dir="results/Turin-128C"
test_name="10162025-${model_name}-${engine}"
results_file_prefix="10162025-${model_name}"

# run the benchmark client
echo "Running benchmark client..."
n_samples=2000

results_dir="${output_dir}/${test_name}"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
fi

for bs in 1 2 4 8; do
    for rep in 1 2 3 4; do
        output_name="${results_file_prefix}-bs${bs}-rep${rep}"
        python main.py run --samples $n_samples --batch-size $bs --output $output_name
        # move results files to results directory. these files start with output_name
        echo "Moving results files to ${results_dir}/ \n"
        mv ./results/${output_name}* "${results_dir}/"
    done
done