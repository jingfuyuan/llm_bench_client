# LLM Benchmark Client (currently works for benchmarking VLM using the COCO dataset)

A comprehensive benchmarking suite for Large Language Models (LLMs), with special focus on Vision-Language Models (VLMs). This tool provides detailed performance metrics, system resource monitoring, and comprehensive reporting capabilities.

## Features

- **Comprehensive Benchmarking**: Measures request latency, time-to-first-token (TTFT), tokens per second, and success rates
- **Dataset Support**: Load datasets from Parquet files with configurable column mappings. Currently supports COCO caption dataset.
- **System Monitoring**: Real-time CPU, memory, and GPU utilization tracking. **Currently, GPU monitoring does not work.**
- **Multiple Output Formats**: Results in JSON, CSV, HTML reports, and text summaries
- **Flexible Configuration**: YAML-based configuration with validation
- **Async Processing**: High-performance asynchronous request handling
- **CLI Interface**: Easy-to-use command-line interface

## Setup conda environment

Create and activate a new conda environment:

```bash
conda env create -f environment.yaml
conda activate llm_bench_client
```

### Optional GPU Monitoring (not tested yet)

For GPU monitoring capabilities, install the optional dependency:

```bash
pip install py3nvml
```

## Dataset for benchmarking VLMs
This benchmark is designed to work with the COCO caption dataset stored in Parquet format. The dataset can be downloaded from huggingface.  
Here is the link: [lmms-lab/COCO-Caption](https://huggingface.co/datasets/lmms-lab/COCO-Caption)


## Quick Start

### 1. Create Configuration (optional)
You don't have to create a configuration file. The benchmark can run with default settings. You can modify `config/default_config.yaml` as needed. 

Create a configuration file for your setup:

```bash
python main.py create-config --output config/my_config.yaml
```

### 2. Edit Configuration

Edit the generated configuration file to match your setup:

```yaml
# Server Configuration
server:
  url: "http://localhost:8000"  # Your LLM server URL

# Dataset Configuration  
dataset:
  path: "/path/to/your/dataset"
  file_pattern: "*.parquet"
  columns:
    image: "image"
    prompt: "question"
    metadata: "question_id"
  image_format: "bytes"
  custom_prompt: null    # Custom prompt template (if any)

# Benchmark Configuration
benchmark:
  num_samples: 100
  max_tokens: 512
  temperature: 0.8
  batch_size: 1
  timeout: 120
```

### 3. Preview Dataset

Verify your dataset is configured correctly:

```bash
python main.py preview --config my_config.yaml --index 0
```

### 4. Run Benchmark

Ommitting the `--config` flag will use the default configuration file located at `config/default_config.yaml`. 
`--samples` and `--batch-size` flags will override the corresponding values in the configuration file.

Execute the benchmark:

```bash
python main.py run --config my_config.yaml --samples 50
```

## Configuration

The configuration file supports the following sections:

### Server Configuration

```yaml
server:
  url: "http://localhost:8000"  # LLM server endpoint
```

### Dataset Configuration

```yaml
dataset:
  path: "/path/to/dataset"      # Path to parquet files
  file_pattern: "*.parquet"     # File pattern to match
  columns:                      # Column name mappings
    image: "image"              # Column containing image data
    prompt: "question"          # Column containing prompts
    metadata: "question_id"     # Column containing metadata
  image_format: "bytes"         # Image format: "bytes", "base64", "path"
```

### Benchmark Configuration

```yaml
benchmark:
  num_samples: 100              # Number of samples to test (0 = all)
  max_tokens: 512               # Maximum output tokens
  temperature: 0.8              # Generation temperature
  top_p: 0.95                   # Top-p sampling
  batch_size: 1                 # Concurrent requests
  timeout: 120                  # Request timeout (seconds)
  warmup_requests: 5            # Warmup requests before benchmark
  batch_delay: 0.5              # Delay between batches
```

### Output Configuration

```yaml
output:
  results_dir: "./results"      # Output directory
  formats:                      # Output formats
    - "json"
    - "csv" 
    - "html"
  include_responses: true       # Include full responses in output
  include_metrics: true         # Include performance metrics
  include_system_metrics: true  # Include system resource metrics
```

### Monitoring Configuration

```yaml
monitoring:
  enabled: true                 # Enable system monitoring
  interval: 1.0                 # Monitoring interval (seconds)
  metrics:                      # Metrics to collect
    - "cpu_percent"
    - "memory_percent"
    - "gpu_utilization"
    - "gpu_memory"
```

## CLI Usage

### Run Benchmark

```bash
# Basic usage
python main.py run

# With custom config and parameters
python main.py run --config my_config.yaml --samples 100 --batch-size 2

# Custom output name
python main.py run --output my_experiment_v1
```

### Preview Dataset

```bash
# Preview first sample
python main.py preview --config my_config.yaml

# Preview specific sample
python main.py preview --config my_config.yaml --index 5
```

### Create Configuration

```bash
python main.py create-config --output new_config.yaml
```

## Output Files

The benchmark generates several output files:

- **`{run_name}_results.json`**: Complete results in JSON format
- **`{run_name}_requests.csv`**: Per-request metrics in CSV format
- **`{run_name}_system_metrics.csv`**: System resource usage over time
- **`{run_name}_summary.csv`**: Summary statistics
- **`{run_name}_report.html`**: Interactive HTML report
- **`{run_name}_summary.txt`**: Text summary

## Metrics Collected

### Request Metrics

- Request latency (total processing time)
- Time to first token (TTFT)
- Tokens per second
- Token counts (prompt, completion, total)
- Success/failure status
- Error messages (if any)

### System Metrics

- CPU utilization percentage
- Memory usage percentage and absolute values
- GPU utilization percentage (if available)
- GPU memory usage (if available)


## Troubleshooting

### Common Issues

1. **Server Connection Failed**
   - Verify the server URL in your configuration
   - Ensure the LLM server is running and accessible
   - Check firewall settings

2. **Dataset Not Found**
   - Verify the dataset path exists
   - Check file pattern matches your files
   - Ensure proper file permissions

3. **GPU Monitoring Not Working**
   - Install py3nvml: `pip install py3nvml`
   - Ensure NVIDIA drivers are installed
   - Check GPU compatibility
