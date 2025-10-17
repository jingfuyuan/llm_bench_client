"""Results management and export utilities."""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .config import Config


class ResultsManager:
    """Manages benchmark results storage and export."""
    
    def __init__(self, config: Config):
        """Initialize results manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.results_dir = Path(config.get('output.results_dir', './results'))
        self.output_formats = config.get('output.formats', ['json', 'csv'])
        self.include_responses = config.get('output.include_responses', True)
        self.include_metrics = config.get('output.include_metrics', True)
        self.include_system_metrics = config.get('output.include_system_metrics', True)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], run_name: Optional[str] = None) -> Dict[str, str]:
        """Save benchmark results in configured formats.
        
        Args:
            results: Complete benchmark results
            run_name: Optional run name for file naming
            
        Returns:
            Dictionary mapping format names to file paths
        """
        if run_name is None:
            run_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        saved_files = {}
        
        for format_type in self.output_formats:
            try:
                if format_type == 'json':
                    file_path = self._save_json(results, run_name)
                elif format_type == 'csv':
                    file_path = self._save_csv(results, run_name)
                elif format_type == 'html':
                    file_path = self._save_html(results, run_name)
                else:
                    self.logger.warning(f"Unsupported output format: {format_type}")
                    continue
                
                saved_files[format_type] = file_path
                self.logger.info(f"Saved {format_type} results to: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save {format_type} results: {e}")
        
        # Save summary file
        summary_path = self._save_summary(results, run_name)
        saved_files['summary'] = summary_path
        
        return saved_files
    
    def _save_json(self, results: Dict[str, Any], run_name: str) -> str:
        """Save results as JSON file.
        
        Args:
            results: Results to save
            run_name: Run name for file naming
            
        Returns:
            Path to saved file
        """
        file_path = self.results_dir / f"{run_name}_results.json"
        
        # Prepare data for JSON serialization
        json_data = results.copy()
        
        # Filter out responses if not requested
        if not self.include_responses and 'request_metrics' in json_data:
            for metric in json_data['request_metrics']:
                metric.pop('response_text', None)
                metric.pop('server_response', None)
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return str(file_path)
    
    def _save_csv(self, results: Dict[str, Any], run_name: str) -> str:
        """Save results as CSV files.
        
        Args:
            results: Results to save
            run_name: Run name for file naming
            
        Returns:
            Path to main CSV file
        """
        # Save request metrics
        if 'request_metrics' in results and results['request_metrics']:
            request_df = pd.DataFrame(results['request_metrics'])
            
            # Filter columns based on configuration
            if not self.include_responses:
                request_df = request_df.drop(['response_text', 'server_response'], 
                                           axis=1, errors='ignore')
            
            request_csv_path = self.results_dir / f"{run_name}_requests.csv"
            request_df.to_csv(request_csv_path, index=False)
        
        # Save system metrics
        if (self.include_system_metrics and 'system_metrics' in results 
            and results['system_metrics']):
            system_df = pd.DataFrame(results['system_metrics'])
            system_csv_path = self.results_dir / f"{run_name}_system_metrics.csv"
            system_df.to_csv(system_csv_path, index=False)
        
        # Save summary
        if 'summary' in results:
            summary_csv_path = self.results_dir / f"{run_name}_summary.csv"
            
            # Flatten summary dictionary
            flattened_summary = self._flatten_dict(results['summary'])
            summary_df = pd.DataFrame([flattened_summary])
            summary_df.to_csv(summary_csv_path, index=False)
        
        return str(request_csv_path) if 'request_metrics' in results else str(summary_csv_path)
    
    def _save_html(self, results: Dict[str, Any], run_name: str) -> str:
        """Save results as HTML report.
        
        Args:
            results: Results to save
            run_name: Run name for file naming
            
        Returns:
            Path to saved HTML file
        """
        file_path = self.results_dir / f"{run_name}_report.html"
        
        html_content = self._generate_html_report(results, run_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(file_path)
    
    def _save_summary(self, results: Dict[str, Any], run_name: str) -> str:
        """Save summary text file.
        
        Args:
            results: Results to save
            run_name: Run name for file naming
            
        Returns:
            Path to saved summary file
        """
        file_path = self.results_dir / f"{run_name}_summary.txt"
        
        summary_text = self._generate_summary_text(results, run_name)
        
        with open(file_path, 'w') as f:
            f.write(summary_text)
        
        return str(file_path)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _generate_html_report(self, results: Dict[str, Any], run_name: str) -> str:
        """Generate HTML report.
        
        Args:
            results: Results data
            run_name: Run name
            
        Returns:
            HTML content string
        """
        summary = results.get('summary', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Benchmark Report - {run_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VLM Benchmark Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p class="timestamp">Run: {run_name}</p>
        
        <h2>Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_samples', 'N/A')}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('successful_requests', 'N/A')}</div>
                <div class="metric-label">Successful Requests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('success_rate', 0) * 100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('average_request_time', 0):.2f}s</div>
                <div class="metric-label">Avg Request Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('average_ttft', 0):.2f}s</div>
                <div class="metric-label">Avg Time to First Token</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('average_tokens_per_second', 0):.1f}</div>
                <div class="metric-label">Avg Tokens/Second</div>
            </div>
        </div>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Average</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            <tr>
                <td>Request Time (s)</td>
                <td>{summary.get('average_request_time', 0):.3f}</td>
                <td>{summary.get('min_request_time', 0):.3f}</td>
                <td>{summary.get('max_request_time', 0):.3f}</td>
            </tr>
            <tr>
                <td>Time to First Token (s)</td>
                <td>{summary.get('average_ttft', 0):.3f}</td>
                <td>{summary.get('min_ttft', 0):.3f}</td>
                <td>{summary.get('max_ttft', 0):.3f}</td>
            </tr>
            <tr>
                <td>Tokens per Second</td>
                <td>{summary.get('average_tokens_per_second', 0):.1f}</td>
                <td>{summary.get('min_tokens_per_second', 0):.1f}</td>
                <td>{summary.get('max_tokens_per_second', 0):.1f}</td>
            </tr>
        </table>
        """
        
        # Add system resources if available
        if 'system_resources' in summary:
            sys_res = summary['system_resources']
            html += f"""
        <h2>System Resources</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{sys_res.get('average_cpu_percent', 0):.1f}%</div>
                <div class="metric-label">Avg CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sys_res.get('average_memory_percent', 0):.1f}%</div>
                <div class="metric-label">Avg Memory Usage</div>
            </div>
            """
            
            if 'average_gpu_utilization' in sys_res:
                html += f"""
            <div class="metric-card">
                <div class="metric-value">{sys_res.get('average_gpu_utilization', 0):.1f}%</div>
                <div class="metric-label">Avg GPU Utilization</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sys_res.get('average_gpu_memory_gb', 0):.1f} GB</div>
                <div class="metric-label">Avg GPU Memory</div>
            </div>
            """
            
            html += "</div>"
        
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_summary_text(self, results: Dict[str, Any], run_name: str) -> str:
        """Generate text summary.
        
        Args:
            results: Results data
            run_name: Run name
            
        Returns:
            Summary text
        """
        summary = results.get('summary', {})
        
        text = f"""VLM Benchmark Summary - {run_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== OVERALL RESULTS ===
Total Samples: {summary.get('total_samples', 'N/A')}
Successful Requests: {summary.get('successful_requests', 'N/A')}
Failed Requests: {summary.get('failed_requests', 'N/A')}
Success Rate: {summary.get('success_rate', 0) * 100:.1f}%
Total Time: {summary.get('total_time', 0):.2f}s

=== PERFORMANCE METRICS ===
Average Request Time: {summary.get('average_request_time', 0):.3f}s
Min Request Time: {summary.get('min_request_time', 0):.3f}s
Max Request Time: {summary.get('max_request_time', 0):.3f}s

Average Time to First Token: {summary.get('average_ttft', 0):.3f}s
Min Time to First Token: {summary.get('min_ttft', 0):.3f}s
Max Time to First Token: {summary.get('max_ttft', 0):.3f}s

Average Tokens per Second: {summary.get('average_tokens_per_second', 0):.1f}
Min Tokens per Second: {summary.get('min_tokens_per_second', 0):.1f}
Max Tokens per Second: {summary.get('max_tokens_per_second', 0):.1f}
"""
        
        # Add system resources if available
        if 'system_resources' in summary:
            sys_res = summary['system_resources']
            text += f"""
=== SYSTEM RESOURCES ===
Average CPU Usage: {sys_res.get('average_cpu_percent', 0):.1f}%
Max CPU Usage: {sys_res.get('max_cpu_percent', 0):.1f}%
Average Memory Usage: {sys_res.get('average_memory_percent', 0):.1f}%
Max Memory Usage: {sys_res.get('max_memory_percent', 0):.1f}%
"""
            
            if 'average_gpu_utilization' in sys_res:
                text += f"""Average GPU Utilization: {sys_res.get('average_gpu_utilization', 0):.1f}%
Max GPU Utilization: {sys_res.get('max_gpu_utilization', 0):.1f}%
Average GPU Memory: {sys_res.get('average_gpu_memory_gb', 0):.1f} GB
Max GPU Memory: {sys_res.get('max_gpu_memory_gb', 0):.1f} GB
"""
        
        return text
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load results from JSON file.
        
        Args:
            file_path: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def compare_results(self, result_files: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark results.
        
        Args:
            result_files: List of result file paths
            
        Returns:
            Comparison summary
        """
        results = []
        for file_path in result_files:
            result = self.load_results(file_path)
            result['file_path'] = file_path
            results.append(result)
        
        comparison = {
            'runs': len(results),
            'comparisons': []
        }
        
        for i, result in enumerate(results):
            summary = result.get('summary', {})
            comparison['comparisons'].append({
                'run_name': Path(result['file_path']).stem,
                'success_rate': summary.get('success_rate', 0),
                'avg_request_time': summary.get('average_request_time', 0),
                'avg_ttft': summary.get('average_ttft', 0),
                'avg_tokens_per_second': summary.get('average_tokens_per_second', 0),
                'total_samples': summary.get('total_samples', 0)
            })
        
        return comparison
