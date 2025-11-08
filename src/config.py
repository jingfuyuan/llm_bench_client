"""Configuration management for VLM benchmarking."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the benchmarking suite."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = self._find_default_config()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        # self._validate_config()
    
    def _find_default_config(self) -> str:
        """Find the default configuration file."""
        current_dir = Path(__file__).parent.parent
        config_file = current_dir / "config" / "default_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Default config file not found: {config_file}")
        
        return str(config_file)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Configuration file is empty or invalid")
            return config
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML in config file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['server', 'dataset', 'benchmark', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate server URL
        server_url = self.config.get('server', {}).get('url')
        if not server_url:
            raise ValueError("Missing required config: server.url")
        
        # Validate dataset path
        dataset_path = self.config.get('dataset', {}).get('path')
        if not dataset_path:
            raise ValueError("Missing required config: dataset.path")
        elif not dataset_path.startswith('/path/to/') and not os.path.exists(dataset_path):
            print(f"Warning: Dataset path does not exist: {dataset_path}")
        
        # Validate dataset columns
        columns = self.config.get('dataset', {}).get('columns', {})
        required_columns = ['image', 'prompt']
        for col in required_columns:
            if col not in columns:
                raise ValueError(f"Missing required dataset column mapping: {col}")
        
        # Validate benchmark settings
        num_samples = self.config.get('benchmark', {}).get('num_samples', 0)
        if not isinstance(num_samples, int) or num_samples < 0:
            raise ValueError("benchmark.num_samples must be a non-negative integer")
        
        max_tokens = self.config.get('benchmark', {}).get('max_tokens', 512)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("benchmark.max_tokens must be a positive integer")
        
        timeout = self.config.get('benchmark', {}).get('timeout', 120)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("benchmark.timeout must be a positive number")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'server.url')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file.
        
        Args:
            path: Path to save config. If None, uses original path.
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {save_path}: {e}")
    
    def create_user_config(self, user_config_path: str) -> str:
        """Create a user-specific configuration file.
        
        Args:
            user_config_path: Path for the new user config
            
        Returns:
            Path to the created config file
        """
        user_config = self.config.copy()
        
        # Update with user-specific defaults
        # user_config['output']['results_dir'] = f"./results_{Path(user_config_path).stem}"
        user_config_path = Path(user_config_path).absolute()

        try:
            user_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(user_config_path, 'w') as f:
                yaml.dump(user_config, f, default_flow_style=False, indent=2)
            
            return user_config_path
        except Exception as e:
            raise RuntimeError(f"Failed to create user config at {user_config_path}: {e}")
    
    def validate_runtime_dependencies(self) -> Dict[str, bool]:
        """Validate that runtime dependencies are available.
        
        Returns:
            Dictionary of dependency status
        """
        status = {}
        
        # Check if dataset path exists
        dataset_path = self.get('dataset.path')
        status['dataset_path_exists'] = os.path.exists(dataset_path) if dataset_path else False
        
        # Check if output directory can be created
        output_dir = self.get('output.results_dir', './results')
        try:
            os.makedirs(output_dir, exist_ok=True)
            status['output_dir_writable'] = True
        except Exception:
            status['output_dir_writable'] = False
        
        return status
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config(path={self.config_path}, sections={list(self.config.keys())})"