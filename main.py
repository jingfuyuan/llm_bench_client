#!/usr/bin/env python3
"""Main entry point for the LLM benchmarking client."""

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.config import Config
from src.benchmark_client import BenchmarkClient
from src.results_manager import ResultsManager
from src.dataset_loader import DatasetLoader


def setup_logging(config: Config):
    """Setup logging configuration."""
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.file', './logs/benchmark.log')
    console = config.get('logging.console', True)
    log_format = config.get('logging.format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)


def validate_server_connection(config: Config) -> bool:
    """Validate that the server is accessible."""
    import requests
    
    server_url = config.get('server.url', 'http://localhost:8080')
    
    try:
        # Try to reach the server health endpoint
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to server at {server_url}: {e}")
        return False


async def run_benchmark(config_path: Optional[str] = None, 
                       num_samples: Optional[int] = None,
                       batch_size: int = 1,
                       output_name: Optional[str] = None) -> bool:
    """Run the benchmark with given parameters.
    
    Args:
        config_path: Path to configuration file
        num_samples: Number of samples to test
        batch_size: Batch size for concurrent requests
        output_name: Name for output files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        config = Config(config_path)
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting LLM benchmark")
        logger.info(f"Configuration loaded from: {config.config_path}")
        
        # Validate server connection
        if not validate_server_connection(config):
            logger.error("Server validation failed. Please ensure the LLM server is running.")
            return False
        
        # Override config parameters if provided
        if num_samples is not None:
            config.set('benchmark.num_samples', num_samples)
        if batch_size != 1:
            config.set('benchmark.batch_size', batch_size)
        
        # Initialize dataset loader to validate dataset
        dataset_loader = DatasetLoader(config)
        # dataset_info = dataset_loader.get_dataset_info()
        # logger.info(f"Dataset loaded: {dataset_info['total_samples']} samples from {dataset_info['num_files']} files")
        
        # Initialize benchmark client
        benchmark_client = BenchmarkClient(config)
        
        # Run benchmark
        results = await benchmark_client.run_benchmark(
            num_samples=config.get('benchmark.num_samples'),
            batch_size=config.get('benchmark.batch_size', 1)
        )
        
        # Get detailed results
        detailed_results = benchmark_client.get_detailed_results()
        detailed_results['summary'] = results
        
        # Save results
        results_manager = ResultsManager(config)
        saved_files = results_manager.save_results(detailed_results, output_name)
        
        # Print summary
        print("\n" + "="*50)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Total Samples: {results['total_samples']}")
        print(f"Successful Requests: {results['successful_requests']}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Average Request Time: {results['average_request_time']:.3f}s")
        print(f"requests/second: {results['requests_per_second']:.3f}")
        print(f'Average Output Length: {results.get("average_output_length", 0):.1f} tokens')
        if 'average_ttft' in results:
            print(f"Average TTFT: {results['average_ttft']:.3f}s")
        if 'average_tokens_per_second' in results:
            print(f"Average Tokens/Second: {results['average_tokens_per_second']:.1f}")
        print(f"Total Time: {results['total_time']:.2f}s")
        
        print("\nResults saved to:")
        for format_type, file_path in saved_files.items():
            print(f"  {format_type}: {file_path}")
        
        logger.info("Benchmark completed successfully")
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"Error: {e}")
        return False


def preview_dataset(config_path: Optional[str] = None, index: int = 0):
    """Preview a sample from the dataset."""
    try:
        config = Config(config_path)
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        dataset_loader = DatasetLoader(config)
        
        # Get dataset info
        info = dataset_loader.get_dataset_info()
        print(f"Dataset Info:")
        print(f"  Files: {info['num_files']}")
        print(f"  Total Samples: {info['total_samples']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Image Format: {info['image_format']}")
        
        # Preview sample
        sample = dataset_loader.preview_sample(index)
        print(f"\nSample {index}:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Image Size: {sample['image_size_bytes']} bytes")
        print(f"  Image Dimensions: {sample['image_dimensions']}")
        print(f"  Image Mode: {sample['image_mode']}")
        if sample['metadata']:
            print(f"  Metadata: {sample['metadata']}")
        
    except Exception as e:
        print(f"Error previewing dataset: {e}")


def create_config(output_path: str):
    """Create a new configuration file."""
    try:
        # Load default config
        default_config = Config()
        
        # Create user config
        user_config_path = default_config.create_user_config(output_path)
        print(f"Configuration file created at: {user_config_path}")
        print("Please edit the configuration file to match your setup.")
        
    except Exception as e:
        print(f"Error creating config: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="LLM Benchmarking Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config
  python main.py run
  
  # Run with custom config and parameters
  python main.py run --config my_config.yaml --samples 50 --batch-size 2
  
  # Preview dataset
  python main.py preview --config my_config.yaml --index 0
  
  # Create new config file
  python main.py create-config --output my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run benchmark command
    run_parser = subparsers.add_parser('run', help='Run benchmark')
    run_parser.add_argument('--config', '-c', type=str, 
                           help='Path to configuration file')
    run_parser.add_argument('--samples', '-n', type=int,
                           help='Number of samples to test')
    run_parser.add_argument('--batch-size', '-b', type=int, default=1,
                           help='Batch size for concurrent requests')
    run_parser.add_argument('--output', '-o', type=str,
                           help='Output name for result files')
    
    # Preview dataset command
    preview_parser = subparsers.add_parser('preview', help='Preview dataset')
    preview_parser.add_argument('--config', '-c', type=str,
                               help='Path to configuration file')
    preview_parser.add_argument('--index', '-i', type=int, default=0,
                               help='Sample index to preview')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration file')
    config_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output path for configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'run':
        success = asyncio.run(run_benchmark(
            config_path=args.config,
            num_samples=args.samples,
            batch_size=args.batch_size,
            output_name=args.output
        ))
        sys.exit(0 if success else 1)
    
    elif args.command == 'preview':
        preview_dataset(args.config, args.index)
    
    elif args.command == 'create-config':
        create_config(args.output)


if __name__ == '__main__':
    main()