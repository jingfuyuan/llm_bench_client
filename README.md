# VLM Benchmark Client

A client-side benchmarking tool for measuring the performance of Vision-Language Models (VLMs) using the COCO caption dataset. This tool sends requests to a VLM server and collects detailed performance metrics including latency, throughput, and resource utilization.

> **⚠️ Important**: This is a **client-side** benchmarking tool. You must have a VLM server running and accessible before using this package. The benchmark client sends image+text requests to the server and measures response times and quality.

## Features

- **Comprehensive Performance Metrics**: Measures request latency, time-to-first-token (TTFT), tokens per second, and success rates
- **Dataset Support**: Works with COCO caption dataset in Parquet format with configurable column mappings
- **System Monitoring**: Real-time CPU and memory utilization tracking (GPU monitoring available but not fully tested)
- **Multiple Output Formats**: Generates results in JSON, CSV, HTML reports, and text summaries
- **Flexible Configuration**: YAML-based configuration with sensible defaults
- **Async Processing**: High-performance asynchronous request handling for concurrent requests
- **Batch Testing**: Built-in support for sweeping different batch sizes

## Prerequisites

Before running the benchmark, ensure you have:

1. **A running VLM server** that supports:
   - OpenAI-compatible API endpoint: `/v1/chat/completions`
   - Health check endpoint: `/health` (used by the client to verify server connectivity)
   - Streaming responses with `stream_options: {"include_usage": true}`. This is included in the HTTP request payload. See script `src/benchmark_client.py`.
   
   Compatible servers include:
   - **vLLM** with vision model support
   - **llama.cpp** server with multimodal (mmproj) support
   - Any **OpenAI-compatible** VLM server

2. **COCO Caption dataset** - Download from [lmms-lab/COCO-Caption](https://huggingface.co/datasets/lmms-lab/COCO-Caption) on Hugging Face.

3. **Python 3.8+** with conda (recommended)

## Installation

### Create and Activate Conda Environment

```bash
conda env create -f conda_env.yaml
conda activate vlm_bench_env
```

## Quick Start

### Step 1: Start Your VLM Server

**The VLM server must be running before you execute the benchmark.** The client will first check the server's `/health` endpoint to verify connectivity.

Example: Starting a llama.cpp server with vision support with minimum arguments. Include other appropriate arguments for your specific tests.
```bash
llama-server --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf --host 0.0.0.0 --port 8000
```

Example: Starting vLLM with a vision model wtih minimum arguments. Include other appropriate arguments for your specific tests.
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000
```

### Step 2: Configure the Benchmark

Edit `config/default_config.yaml` to match your setup:

```yaml
# Server Configuration - Update this to your VLM server URL
server:
  url: "http://localhost:8000"

# Dataset Configuration - Update path to your COCO dataset
dataset:
  # Path to the parquet files directory
  path: "/home/amd/workspace/vlm/coco_captions/data"
  # Parquet file pattern (glob)
  file_pattern: "val-00000-of-00013.parquet"
  # Column names in the parquet files
  columns:
    image: "image"          # Column containing image bytes
    prompt: "question"        # Column containing text prompts
    metadata: "question_id"    # Column containing additional metadata
  # Image format (if images need conversion)
  image_format: "bytes"     # "bytes", "base64", "path"
  custom_prompt: null    # Custom prompt template (if any)

# Benchmark Configuration
benchmark:
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  num_samples: 100
  max_tokens: 512
  temperature: 0.8
  batch_size: 1
  timeout: 120
  warmup_requests: 10
```

Or create a custom configuration file:

```bash
python main.py create-config --output config/my_config.yaml
```

### Step 3: Verify Dataset Configuration

Preview a sample to ensure the dataset is configured correctly:

```bash
python main.py preview --config config/default_config.yaml --index 0
```

### Step 4: Run the Benchmark

```bash
# Run with default configuration
python main.py run

# Run with custom parameters
python main.py run --samples 100 --batch-size 2

# Run with custom config and output name
python main.py run --config config/my_config.yaml --samples 50 --output my_experiment
```
You can use command line arguments to override the setting in the config file.

## Batch Testing (Sweep Different Batch Sizes)

The `batch_test.sh` script allows you to run benchmarks across multiple batch sizes for performance comparison:

```bash
# Edit batch_test.sh to customize:
# - model_name: Name for output files
# - n_samples: Number of samples per test
# - batch sizes to test (default: 1, 2, 4, 8, 16)

# Run the batch test
./batch_test.sh
```

This script will:
1. Run benchmarks for each specified batch size
2. Organize results into a dedicated output directory
3. Name output files with model name, batch size, and repetition number

## CLI Reference

### Run Benchmark

```text
python main.py run [OPTIONS]

Options:
  --config, -c PATH      Path to configuration file (default: config/default_config.yaml)
  --samples, -n INT      Number of samples to test (overrides config)
  --batch-size, -b INT   Concurrent requests / batch size (default: 1). Overrides config
  --output, -o NAME      Output name prefix for result files
```

### Preview Dataset

```text
python main.py preview [OPTIONS]

Options:
  --config, -c PATH      Path to configuration file
  --index, -i INT        Sample index to preview (default: 0)
```

### Create Configuration

```bash
python main.py create-config --output PATH
```

## Configuration Reference

### Server Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `server.url` | VLM server endpoint URL | `http://localhost:8000` |

### Dataset Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset.path` | Path to parquet files directory | - |
| `dataset.file_pattern` | Glob pattern for parquet files | `*.parquet` |
| `dataset.columns.image` | Column containing image data | `image` |
| `dataset.columns.prompt` | Column containing prompts | `question` |
| `dataset.columns.metadata` | Column containing metadata | `question_id` |
| `dataset.image_format` | Image format: `bytes`, `base64`, `path` | `bytes` |
| `dataset.custom_prompt` | Custom prompt. If null, use the default from COCO dataset | `null` |

### Benchmark Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `benchmark.model_name` | Model identifier (sent in API request) | - |
| `benchmark.num_samples` | Number of samples to test (0 = all) | `100` |
| `benchmark.max_tokens` | Maximum output tokens | `512` |
| `benchmark.temperature` | Generation temperature | `0.8` |
| `benchmark.top_p` | Top-p sampling | `0.95` |
| `benchmark.batch_size` | Concurrent requests | `1` |
| `benchmark.timeout` | Request timeout (seconds) | `120` |
| `benchmark.warmup_requests` | Warmup requests before benchmark | `10` |
| `benchmark.batch_delay` | Delay between batches (seconds) | `0` |
| `benchmark.max_retries` | Retries for failed requests | `3` |

### Output Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output.results_dir` | Output directory | `./results` |
| `output.formats` | Output formats list | `["json", "csv", "html"]` |
| `output.include_responses` | Include full responses | `true` |
| `output.include_metrics` | Include performance metrics | `true` |
| `output.include_system_metrics` | Include system metrics | `true` |

### Monitoring Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.enabled` | Enable system monitoring | `true` |
| `monitoring.interval` | Monitoring interval (seconds) | `1.0` |
| `monitoring.metrics` | Metrics to collect | `["cpu_percent", "memory_percent", "gpu_utilization", "gpu_memory"]` |

## Output Files

Each benchmark run generates the following files in the `results/` directory:

| File | Description |
|------|-------------|
| `{name}_results.json` | Complete results with all metrics in JSON format |
| `{name}_requests.csv` | Per-request metrics (latency, TTFT, tokens, etc.) |
| `{name}_system_metrics.csv` | System resource usage over time |
| `{name}_summary.csv` | Summary statistics in CSV format |
| `{name}_summary.txt` | Human-readable text summary |
| `{name}_report.html` | Interactive HTML report with visualizations |

## Understanding the Results

### Console Output

After a successful benchmark run, you'll see a summary like:

```
==================================================
BENCHMARK COMPLETED SUCCESSFULLY
==================================================
Total Samples: 100
Successful Requests: 98
Success Rate: 98.0%
Average Request Time: 2.345s
requests/second: 4.26
Average Output Length: 45.2 tokens
Average TTFT: 0.523s
Average Tokens/Second: 35.8
Total Time: 23.00s
```

### Key Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Success Rate** | Percentage of requests completed without errors | Higher is better; <95% may indicate server issues |
| **Average Request Time** | Total time from request sent to response received | Lower is better; includes network + processing time |
| **requests/second** | Throughput of the system | Higher is better |
| **Average TTFT** | Time to First Token - latency before generation starts | Lower is better; critical for interactive applications |
| **Average Tokens/Second** | Token generation speed (completion tokens / generation time) | Higher is better |
| **Average Output Length** | Average number of tokens in model responses | Varies by prompt and max_tokens setting |

### Analyzing Results Files

#### Per-Request Analysis (`_requests.csv`)

Contains detailed metrics for each request:
- `request_id`: Unique identifier for the request
- `question_id`: Original dataset sample ID
- `time_to_first_token`: TTFT in seconds
- `total_processing_time`: End-to-end latency
- `tokens_per_second`: Generation speed
- `prompt_tokens`, `completion_tokens`, `total_tokens`: Token counts
- `success`: Whether request succeeded
- `error_message`: Error details if failed
- `response_text`: Model's generated response

Use this file to:
- Identify slow or failed requests
- Analyze latency distribution (compute percentiles)
- Correlate performance with image size or prompt length

#### System Metrics (`_system_metrics.csv`)

Time-series data of resource usage during the benchmark:
- `timestamp`: When the measurement was taken
- `cpu_percent`: CPU utilization
- `memory_percent`, `memory_used_gb`: RAM usage
- `gpu_utilization`, `gpu_memory_used_gb`: GPU metrics (if available)

Use this file to:
- Identify resource bottlenecks
- Monitor memory leaks during long benchmarks
- Correlate system load with request latency

## collect results
Using the data collection script `collect_data.py` to aggregate summary results from multiple benchmark runs into a single CSV file for easy comparison and analysis.

```bash
# Collect all summary files from a results directory
python src/collect_data.py --input_dir ./results --output_file collected_results.csv

# Collect from a specific experiment directory
python src/collect_data.py --input_dir ./results/my_experiment --output_file my_experiment_collected.csv
```

The script:

- Recursively searches for all `*_summary.csv` files in the specified directory
- Extracts `batch_size` and `repeat_number` from filenames matching the pattern `*-bs{batch_size}-rep{repeat_number}_summary.csv`
- Combines all data into a single CSV with metadata columns

## Troubleshooting

### Server Connection Failed

```
Failed to connect to server at http://localhost:8000
```

**Cause**: The benchmark client checks the `/health` endpoint before running.

**Solutions:**
1. Verify the VLM server is running: `curl http://localhost:8000/health`
2. If your server doesn't have a `/health` endpoint, you may need to modify `main.py` to skip the health check or implement the endpoint on your server
3. Check the server URL in your configuration file
4. Ensure no firewall is blocking the connection
5. Verify the server is listening on the correct host/port

### Dataset Not Found

```
Error: Dataset path does not exist
```

**Solutions:**
1. Verify the dataset path exists: `ls /path/to/your/dataset`
2. Check the `file_pattern` matches your parquet files
3. Ensure proper file permissions
