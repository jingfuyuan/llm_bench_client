"""Dataset loading and processing utilities."""

import os
import glob
import base64
from io import BytesIO
from typing import List, Dict, Any, Iterator, Optional, Tuple
import pandas as pd
from PIL import Image
import logging
import json

from .config import Config


class DatasetLoader:
    """Loads and processes datasets from parquet files."""
    
    def __init__(self, config: Config):
        """Initialize dataset loader.
        
        Args:
            config: Configuration object
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If dataset path doesn't exist
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.data_name = config.get('dataset.name', None)
        self.dataset_path = config.get('dataset.path')
        self.input_length = config.get('dataset.input_length', 128)
        self.data_file = os.path.join(self.dataset_path, f"prompts_{self.input_length}.txt")
    
    def _validate_dataset_path(self):
        """Validate that dataset path exists and is accessible."""
        if not self.dataset_path:
            raise ValueError("Dataset path not configured. Please set 'dataset.path' in config.")
        
        if self.dataset_path.startswith('/path/to/'):
            raise ValueError("Dataset path appears to be a placeholder. Please update the config file with the actual path.")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path is not a directory: {self.dataset_path}")
        
        # Check if directory is readable
        if not os.access(self.dataset_path, os.R_OK):
            raise PermissionError(f"Dataset path is not readable: {self.dataset_path}")
    
    def _validate_image_format(self):
        """Validate image format configuration."""
        valid_formats = ['bytes', 'base64', 'path']
        if self.image_format not in valid_formats:
            raise ValueError(f"Invalid image format '{self.image_format}'. Must be one of: {valid_formats}")
    
    def get_parquet_files(self) -> List[str]:
        """Get list of parquet files in the dataset directory.
        
        Returns:
            List of parquet file paths
            
        Raises:
            FileNotFoundError: If no parquet files are found
        """
        try:
            pattern = os.path.join(self.dataset_path, self.file_pattern)
            files = glob.glob(pattern)
            
            if not files:
                raise FileNotFoundError(f"No parquet files found with pattern: {pattern}")
            
            # Filter out directories and non-readable files
            valid_files = []
            for file_path in files:
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    valid_files.append(file_path)
                else:
                    self.logger.warning(f"Skipping invalid or unreadable file: {file_path}")
            
            if not valid_files:
                raise FileNotFoundError(f"No valid parquet files found in: {self.dataset_path}")
            
            valid_files.sort()  # Ensure consistent ordering
            self.logger.info(f"Found {len(valid_files)} parquet files")
            return valid_files
            
        except Exception as e:
            self.logger.error(f"Error finding parquet files: {e}")
            raise
    
    def load_dataset(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from parquet files.
        
        Args:
            num_samples: Maximum number of samples to load. If None, loads all.
            
        Returns:
            DataFrame containing the dataset
            
        Raises:
            ValueError: If no data is loaded or dataset is invalid
            Exception: If parquet files cannot be read
        """
        data = []
        sample_count = 0
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
                sample_count += 1
                if sample_count >= num_samples:
                    break
        
        df = pd.DataFrame(data)
        return df

    def _validate_columns(self, df: pd.DataFrame):
        """Validate that required columns exist in the dataset.
        
        Args:
            df: Dataset DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        image_col = self.columns.get('image', 'image')
        prompt_col = self.columns.get('prompt', 'prompt')
        
        missing_columns = []
        
        if image_col not in df.columns:
            missing_columns.append(f"image column '{image_col}'")
        
        if prompt_col not in df.columns:
            missing_columns.append(f"prompt column '{prompt_col}'")
        
        if missing_columns:
            available_columns = list(df.columns)
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}. "
                f"Available columns: {available_columns}"
            )
    
    def iterate_samples(self, num_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples.
        
        Args:
            num_samples: Maximum number of samples to iterate over
            
        Yields:
            Dictionary containing sample data
            
        Raises:
            ValueError: If dataset cannot be loaded or processed
        """
        dataset = self.load_dataset(num_samples)
        for idx, row in dataset.iterrows():
            sample = {
                "prompt_id": int(row["id"]),
                "prompt": row["text"]
            }
            yield sample

    
    def _process_image(self, image_data: Any) -> str:
        """Process image data and convert to base64.
        
        Args:
            image_data: Raw image data from dataset
            
        Returns:
            Base64 encoded image string
            
        Raises:
            ValueError: If image format is unsupported
            Exception: If image processing fails
        """
        try:
            if self.image_format == 'bytes':
                # Image data is already in bytes
                if isinstance(image_data, bytes):
                    image_bytes = image_data
                elif hasattr(image_data, '__iter__') and not isinstance(image_data, (str, dict)):
                    # Convert array-like data to bytes
                    image_bytes = bytes(image_data)
                else:
                    raise ValueError(f"Cannot convert image data of type {type(image_data)} to bytes")
                
                # Validate it's a valid image by trying to open it
                try:
                    Image.open(BytesIO(image_bytes))
                except Exception as e:
                    self.logger.warning(f"Image validation failed, proceeding anyway: {e}")
                
                # Convert to base64
                return base64.b64encode(image_bytes).decode('utf-8')
            
            elif self.image_format == 'base64':
                # Image data is already base64 encoded
                if isinstance(image_data, str):
                    # Validate base64
                    try:
                        base64.b64decode(image_data)
                        return image_data
                    except Exception:
                        raise ValueError("Invalid base64 image data")
                else:
                    return base64.b64encode(bytes(image_data)).decode('utf-8')
            
            elif self.image_format == 'path':
                # Image data is a file path
                if not isinstance(image_data, str):
                    raise ValueError("Image path must be a string")
                
                image_path = os.path.join(self.dataset_path, image_data)
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                if not os.access(image_path, os.R_OK):
                    raise PermissionError(f"Image file not readable: {image_path}")
                
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')
            
            else:
                raise ValueError(f"Unsupported image format: {self.image_format}")
        
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
            
        Raises:
            Exception: If dataset cannot be analyzed
        """
        try:
            files = self.get_parquet_files()
            
            # Get basic info from first file
            first_df = pd.read_parquet(files[0])
            
            # Count total samples across all files
            total_samples = 0
            file_info = []
            
            for file_path in files:
                try:
                    df = pd.read_parquet(file_path)
                    file_size = len(df)
                    total_samples += file_size
                    file_info.append({
                        'path': file_path,
                        'samples': file_size,
                        'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                    })
                except Exception as e:
                    self.logger.warning(f"Could not analyze file {file_path}: {e}")
            
            info = {
                'num_files': len(files),
                'total_samples': total_samples,
                'columns': list(first_df.columns),
                'sample_columns': self.columns,
                'image_format': self.image_format,
                'files': files,
                'file_details': file_info,
                'dataset_path': self.dataset_path,
                'file_pattern': self.file_pattern
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            raise
    
    def preview_sample(self, index: int = 0) -> Dict[str, Any]:
        """Preview a sample from the dataset.
        
        Args:
            index: Index of sample to preview
            
        Returns:
            Sample data dictionary
            
        Raises:
            IndexError: If sample index is not available
            ValueError: If dataset cannot be loaded
        """
        try:
            dataset = self.load_dataset(num_samples=index + 1)
            
            if len(dataset) <= index:
                raise IndexError(f"Sample index {index} not available in dataset (size: {len(dataset)})")
            
            image_col = self.columns.get('image', 'image')
            prompt_col = self.columns.get('prompt', 'prompt')
            metadata_col = self.columns.get('metadata', 'metadata')
            
            row = dataset.iloc[index]
            
            sample = {
                'index': index,
                'prompt': str(row[prompt_col])[:200] + "..." if len(str(row[prompt_col])) > 200 else str(row[prompt_col]),
                'metadata': row.get(metadata_col, {}) if metadata_col in dataset.columns else {},
            }
            
            # Try to get image info
            try:
                image_data = row[image_col]
                if isinstance(image_data, dict) and "bytes" in image_data:
                    image_bytes = image_data["bytes"]
                    sample['image_size_bytes'] = len(image_bytes)
                elif isinstance(image_data, (bytes, bytearray)):
                    sample['image_size_bytes'] = len(image_data)
                else:
                    sample['image_size_bytes'] = 'unknown'
                
                # Try to get image dimensions
                if self.image_format == 'bytes':
                    if isinstance(image_data, dict) and "bytes" in image_data:
                        image_bytes = image_data["bytes"]
                    else:
                        image_bytes = bytes(image_data) if not isinstance(image_data, bytes) else image_data
                    
                    img = Image.open(BytesIO(image_bytes))
                    sample['image_dimensions'] = img.size
                    sample['image_mode'] = img.mode
                else:
                    sample['image_dimensions'] = 'unknown'
                    sample['image_mode'] = 'unknown'
                    
            except Exception as e:
                self.logger.warning(f"Could not analyze image for sample {index}: {e}")
                sample['image_size_bytes'] = 'error'
                sample['image_dimensions'] = 'error'
                sample['image_mode'] = 'error'
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error previewing sample {index}: {e}")
            raise