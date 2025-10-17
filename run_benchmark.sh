#!/bin/bash

# This script will start the llama.cpp server and run benchmarks.

# source the AOCC environment
source /opt/AMD/aocc-compiler-5.0.0/setenv_AOCC.sh

# build the llama.cpp command
llama_server="/home/amd/workspace/vlm/llama.cpp/build/bin/llama-server"
model_path="/home/amd/dataset/hf_home/gguf/Qwen2.5-VL-7B-Instruct-bf16.gguf"
mmproject_path="/home/amd/dataset/hf_home/gguf/Qwen2.5-VL-7B-Instruct-mmproj-bf16.gguf"
host="0.0.0.0"
port="8000"
n_parallel="4"

# start the llama.cpp server in the background
echo "Starting LLaMA server..."
$llama_server --model "$model_path" --mmproj "$mmproject_path" --host "$host" --port "$port" \
    -c 8192 --parallel "$n_parallel" -fa "on" &
server_pid=$!
echo "LLaMA server started with PID $server_pid"

# wait for the server to start
sleep 20

# run the benchmark client
echo "Running benchmark client..."
n_samples=200
for bs in 1 2 4; do
    output_name="10162025-qwen2.5-vl-7b-Mem6400-inst1-bs${bs}"
    python main.py run --samples $n_samples --batch-size $bs --output $output_name
done

# stop the llama.cpp server
echo "Stopping LLaMA server..."
kill $server_pid
echo "LLaMA server stopped."

