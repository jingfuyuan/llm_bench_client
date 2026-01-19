#!/bin/bash

# This script runs the benchmark client with different batch sizes and repetitions.
# Before running this script, ensure that the server is up and running.
# Check the config/default_config.yaml for server URL and other configurations.
# Activate the virtual environment if needed

model_name="llama-3.1-8B-Instruct"
engine="vllm.aim"
output_dir="results/Turin-128C-vllm-zentorch-112125"
test_name="11212025-${model_name}-${engine}"
results_file_prefix="112012025-${model_name}"

# run the benchmark client
echo "Running benchmark client..."
n_samples=128

results_dir="${output_dir}/${test_name}"
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
fi

for bs in 1 2 4 8; do
    for rep in 1; do
        output_name="${results_file_prefix}-bs${bs}-rep${rep}"
        python main.py run --samples $n_samples --batch-size $bs --output $output_name
        # move results files to results directory. these files start with output_name
        echo "Moving results files to ${results_dir}/ \n"
        mv ./results/${output_name}* "${results_dir}/"
    done
done

# copy this script to the results directory for future reference
cp batch_test.sh "${results_dir}/"
