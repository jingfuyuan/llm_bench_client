"""Benchmark client for testing VLM performance."""

import time
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
import traceback

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from .config import Config
from .dataset_loader import DatasetLoader


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    timestamp_begin: str
    timestamp_end: str
    question_id: str
    prompt: str
    image_dimensions: Tuple[int, int]
    image_processing_time: float
    time_to_first_token: float
    total_processing_time: float
    tokens_per_second: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    true_answer: Optional[str] = None
    error_message: Optional[str] = None
    response_text: Optional[str] = None
    server_response: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


class BenchmarkClient:
    """Client for benchmarking VLM performance."""
    
    def __init__(self, config: Config):
        """Initialize benchmark client.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Server configuration  
        self.base_url = config.get('server.url', 'http://localhost:8080')
        if self.base_url.endswith('/'):
            self.base_url = self.base_url.rstrip('/')
        
        # Benchmark configuration
        self.model_name = config.get('benchmark.model_name', 'vision-language-model')
        self.max_tokens = config.get('benchmark.max_tokens', 512)
        self.output_length = config.get('dataset.output_length', 128)
        self.temperature = config.get('benchmark.temperature', 0.8)
        self.top_p = config.get('benchmark.top_p', 0.95)
        self.timeout = config.get('benchmark.timeout', 120)
        self.max_retries = config.get('benchmark.max_retries', 3)
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(config)
        
        # Results storage
        self.request_metrics: List[RequestMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # GPU monitoring setup
        self.gpu_available = False
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                if self.gpu_available:
                    self.logger.info(f"GPU monitoring enabled for {self.gpu_count} GPUs")
            except Exception as e:
                self.logger.warning(f"GPU monitoring not available: {e}")
    
    async def run_benchmark(self, num_samples: Optional[int] = None, 
                          batch_size: int = 1) -> Dict[str, Any]:
        """Run the complete benchmark.
        
        Args:
            num_samples: Number of samples to test
            batch_size: Number of concurrent requests
            
        Returns:
            Benchmark results summary
        """
        if num_samples is None:
            num_samples = self.config.get('benchmark.num_samples', 100)
        
        if num_samples == 0:
            # Use all available samples
            dataset_info = self.dataset_loader.get_dataset_info()
            num_samples = dataset_info['total_samples']
        
        self.logger.info(f"Starting benchmark with {num_samples} samples, batch size {batch_size}")
        
        # Start system monitoring
        monitor_task = None
        if self.config.get('monitoring.enabled', True):
            monitor_task = asyncio.create_task(self._monitor_system())
        
        try:
            # Run warmup requests
            await self._run_warmup()
            
            # Run main benchmark
            start_time = time.time()
            await self._run_requests(num_samples, batch_size)
            total_time = time.time() - start_time
            
            # Generate summary
            summary = self._generate_summary(total_time, num_samples)
            
            self.logger.info("Benchmark completed successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise
        finally:
            # Stop monitoring
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
    
    async def _run_warmup(self):
        """Run warmup requests to prepare the server."""
        warmup_count = self.config.get('benchmark.warmup_requests', 5)
        if warmup_count <= 0:
            return
        
        self.logger.info(f"Running {warmup_count} warmup requests")
        
        try:
            warmup_samples = list(self.dataset_loader.iterate_samples(warmup_count))
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                tasks = []
                for i, sample in enumerate(warmup_samples):
                    task = self._send_request(session, f"warmup_{i}", sample, is_warmup=True)
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info("Warmup completed")
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")
    
    async def _run_requests(self, num_samples: int, batch_size: int):
        """Run the main benchmark requests.
        
        Args:
            num_samples: Number of samples to process
            batch_size: Batch size for concurrent requests
        """
        samples = list(self.dataset_loader.iterate_samples(num_samples))
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                
                # Create tasks for this batch
                tasks = []
                for j, sample in enumerate(batch):
                    request_id = f"request_{i + j}"
                    task = self._send_request(session, request_id, sample)
                    tasks.append(task)
                
                # Wait for batch to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Delay between batches if configured
                batch_delay = self.config.get('benchmark.batch_delay', 0.0)
                if batch_delay > 0 and i + batch_size < len(samples):
                    await asyncio.sleep(batch_delay)
                
                # Log progress
                completed = min(i + batch_size, len(samples))
                self.logger.info(f"Completed {completed}/{len(samples)} requests")
    
    async def _send_request(self, session: aiohttp.ClientSession, 
                          request_id: str, sample: Dict[str, Any], 
                          is_warmup: bool = False) -> Optional[RequestMetrics]:
        """Send a single request to the server.
        
        Args:
            session: HTTP session
            request_id: Unique request identifier
            sample: Sample data
            is_warmup: Whether this is a warmup request
            
        Returns:
            Request metrics or None if warmup
        """
        start_time = time.time()
        timestamp_begin = time.time()
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": sample['prompt']
                }
            ],
            "max_tokens": self.output_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,  # We'll use streaming to measure time to first token
            "stream_options": {"include_usage": True}
        }
        
        try:
            # Send request
            image_processing_start = time.time()
            
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json", "X-Session-ID": request_id},
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                # Process streaming response
                first_token_time = None
                response_text = ""
                server_response = {}
                usage = {}
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content and first_token_time is None:
                                        first_token_time = time.time()
                                    if content:
                                        response_text += content
                            
                            # Store the timings returned from server
                            if 'timings' in data:
                                server_response = data['timings']

                            if 'usage' in data:
                                usage = data['usage']
                        
                        except json.JSONDecodeError:
                            continue
                
                total_time = time.time() - start_time
                image_processing_time = image_processing_start - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else total_time
                
                # Extract token information
                total_tokens = usage.get('total_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                
                # Calculate tokens per second
                if completion_tokens > 0 and total_time > time_to_first_token:
                    generation_time = total_time - time_to_first_token
                    tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
                else:
                    tokens_per_second = 0
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    timestamp_begin=timestamp_begin,
                    timestamp_end=time.time(),
                    question_id=sample['prompt_id'],
                    image_dimensions=None,
                    prompt=sample['prompt'],
                    image_processing_time=image_processing_time,
                    time_to_first_token=time_to_first_token,
                    total_processing_time=total_time,
                    tokens_per_second=tokens_per_second,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    success=True,
                    true_answer=None,
                    response_text=response_text,
                    server_response=server_response
                )
                
                if not is_warmup:
                    self.request_metrics.append(metrics)
                
                return metrics
        
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Request {request_id} failed: {error_msg}")
            
            metrics = RequestMetrics(
                request_id=request_id,
                timestamp_begin=timestamp_begin,
                timestamp_end=time.time(),
                question_id=sample['prompt_id'],
                image_dimensions=None,
                prompt=sample['prompt'],
                image_processing_time=0,
                time_to_first_token=0,
                total_processing_time=total_time,
                tokens_per_second=0,
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                success=False,
                error_message=error_msg
            )
            
            if not is_warmup:
                self.request_metrics.append(metrics)
            
            return metrics
    
    async def _monitor_system(self):
        """Monitor system resources during benchmark."""
        interval = self.config.get('monitoring.interval', 1.0)
        
        while True:
            try:
                timestamp = datetime.now().isoformat()
                
                # CPU and memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                
                metrics = SystemMetrics(
                    timestamp=timestamp,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_gb=memory_used_gb
                )
                
                # GPU metrics if available
                if self.gpu_available:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
                        gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        metrics.gpu_utilization = gpu_util.gpu
                        metrics.gpu_memory_used_gb = gpu_mem.used / (1024**3)
                        metrics.gpu_memory_total_gb = gpu_mem.total / (1024**3)
                    except Exception as e:
                        self.logger.debug(f"GPU monitoring error: {e}")
                
                self.system_metrics.append(metrics)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def _generate_summary(self, total_time: float, num_samples: int) -> Dict[str, Any]:
        """Generate benchmark summary statistics.
        
        Args:
            total_time: Total benchmark time
            num_samples: Number of samples processed
            
        Returns:
            Summary statistics dictionary
        """
        successful_requests = [m for m in self.request_metrics if m.success]
        failed_requests = [m for m in self.request_metrics if not m.success]
        
        if not successful_requests:
            return {
                'total_samples': num_samples,
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'success_rate': 0.0,
                'total_time': total_time,
                'error': 'All requests failed'
            }
        
        # Calculate statistics
        ttft_times = [m.time_to_first_token for m in successful_requests]
        total_times = [m.total_processing_time for m in successful_requests]
        output_lengths = [m.completion_tokens for m in successful_requests if m.completion_tokens > 0]
        tokens_per_sec = [m.tokens_per_second for m in successful_requests if m.tokens_per_second > 0]
        
        summary = {
            'total_samples': num_samples,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / len(self.request_metrics) if self.request_metrics else 0,
            'total_time': total_time,
            'requests_per_second': len(successful_requests) / total_time if total_time > 0 else 0,
            'average_request_time': sum(total_times) / len(total_times) if total_times else 0,
            'min_request_time': min(total_times) if total_times else 0,
            'max_request_time': max(total_times) if total_times else 0,
            'average_ttft': sum(ttft_times) / len(ttft_times) if ttft_times else 0,
            'min_ttft': min(ttft_times) if ttft_times else 0,
            'max_ttft': max(ttft_times) if ttft_times else 0,
            'average_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'min_output_length': min(output_lengths) if output_lengths else 0,
            'max_output_length': max(output_lengths) if output_lengths else 0,
        }
        
        if tokens_per_sec:
            summary.update({
                'average_tokens_per_second': sum(tokens_per_sec) / len(tokens_per_sec),
                'min_tokens_per_second': min(tokens_per_sec),
                'max_tokens_per_second': max(tokens_per_sec),
            })
        
        # System resource summary
        if self.system_metrics:
            cpu_usage = [m.cpu_percent for m in self.system_metrics]
            memory_usage = [m.memory_percent for m in self.system_metrics]
            
            summary['system_resources'] = {
                'average_cpu_percent': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_percent': max(cpu_usage),
                'average_memory_percent': sum(memory_usage) / len(memory_usage),
                'max_memory_percent': max(memory_usage),
            }
            
            if self.gpu_available and any(m.gpu_utilization is not None for m in self.system_metrics):
                gpu_usage = [m.gpu_utilization for m in self.system_metrics if m.gpu_utilization is not None]
                gpu_memory = [m.gpu_memory_used_gb for m in self.system_metrics if m.gpu_memory_used_gb is not None]
                
                if gpu_usage:
                    summary['system_resources'].update({
                        'average_gpu_utilization': sum(gpu_usage) / len(gpu_usage),
                        'max_gpu_utilization': max(gpu_usage),
                    })
                
                if gpu_memory:
                    summary['system_resources'].update({
                        'average_gpu_memory_gb': sum(gpu_memory) / len(gpu_memory),
                        'max_gpu_memory_gb': max(gpu_memory),
                    })
        
        return summary
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed benchmark results.
        
        Returns:
            Dictionary containing all metrics and results
        """
        return {
            'request_metrics': [asdict(m) for m in self.request_metrics],
            'system_metrics': [asdict(m) for m in self.system_metrics],
            'config': self.config.config
        }