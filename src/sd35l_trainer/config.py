"""
Configuration management for SD3.5L LoRA training.
Handles parameter validation, presets, and configuration file loading.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
import yaml
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class TrainingConfig(BaseModel):
    """Configuration model for LoRA training parameters."""
    
    # Required parameters
    dataset_url: str = Field(..., description="Remote dataset ZIP URL")
    destination: str = Field(..., description="Destination model (username/model-name)")
    trigger_word: str = Field(..., description="Unique trigger word")
    
    # Core training parameters with optimized defaults
    steps: int = Field(1500, ge=1000, le=2000, description="Training steps")
    lora_rank: int = Field(16, ge=4, le=128, description="LoRA rank")
    optimizer: str = Field("prodigy", description="Optimizer (prodigy or adamw8bit)")
    batch_size: int = Field(4, ge=1, le=8, description="Batch size")
    resolution: str = Field("768,1024", description="Resolution setting")
    learning_rate: float = Field(1.0, gt=0, description="Learning rate")
    
    # Advanced parameters
    wandb_project: str = Field("sd3.5_train_replicate", description="Weights & Biases project name")
    wandb_save_interval: int = Field(100, ge=10, description="W&B save interval")
    wandb_sample_interval: int = Field(100, ge=10, description="W&B sample interval")
    caption_dropout_rate: float = Field(0.05, ge=0.0, le=1.0, description="Caption dropout rate")
    cache_latents_to_disk: bool = Field(False, description="Cache latents to disk")
    
    @validator('destination')
    def validate_destination(cls, v):
        """Validate destination format (username/model-name)."""
        if '/' not in v:
            raise ValueError("Destination must be in format 'username/model-name'")
        parts = v.split('/')
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("Destination must be in format 'username/model-name'")
        return v
    
    @validator('trigger_word')
    def validate_trigger_word(cls, v):
        """Validate trigger word is unique and not common."""
        common_words = {
            'dog', 'cat', 'person', 'man', 'woman', 'style', 'art', 'image', 
            'photo', 'picture', 'face', 'portrait', 'character', 'model'
        }
        if v.lower() in common_words:
            raise ValueError(f"Trigger word '{v}' is too common. Use a unique identifier.")
        if len(v.strip()) < 2:
            raise ValueError("Trigger word must be at least 2 characters long")
        return v.strip()
    
    @validator('optimizer')
    def validate_optimizer(cls, v):
        """Validate optimizer choice."""
        valid_optimizers = {'prodigy', 'adamw8bit'}
        if v not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of: {', '.join(valid_optimizers)}")
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v, values):
        """Validate learning rate based on optimizer."""
        optimizer = values.get('optimizer', 'prodigy')
        if optimizer == 'prodigy' and (v < 0.1 or v > 10.0):
            raise ValueError("Learning rate for Prodigy should be between 0.1 and 10.0 (typically 1.0)")
        elif optimizer == 'adamw8bit' and (v < 0.0001 or v > 0.01):
            raise ValueError("Learning rate for adamw8bit should be between 0.0001 and 0.01 (typically 0.0004)")
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format."""
        try:
            parts = v.split(',')
            if len(parts) != 2:
                raise ValueError
            width, height = int(parts[0]), int(parts[1])
            if width < 512 or height < 512 or width > 2048 or height > 2048:
                raise ValueError
        except (ValueError, IndexError):
            raise ValueError("Resolution must be in format 'width,height' with values between 512 and 2048")
        return v


class ConfigManager:
    """Manages configuration loading, validation, and presets."""
    
    PRESETS = {
        "beginner": {
            "optimizer": "prodigy",
            "learning_rate": 1.0,
            "steps": 1500,
            "batch_size": 1,  # Conservative for beginners
            "resolution": "768,1024",
            "lora_rank": 16
        },
        "experienced": {
            "optimizer": "prodigy", 
            "learning_rate": 1.0,
            "steps": 1500,
            "batch_size": 4,  # Faster training
            "resolution": "768,1024",
            "lora_rank": 16
        },
        "fast": {
            "optimizer": "adamw8bit",
            "learning_rate": 0.0004,
            "steps": 1000,
            "batch_size": 4,
            "resolution": "512,768",
            "lora_rank": 16
        }
    }
    
    @classmethod
    def load_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Load a configuration preset."""
        if preset_name not in cls.PRESETS:
            available = ', '.join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def load_config_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            # Flatten nested structure if present
            flattened = {}
            for key, value in config.items():
                if isinstance(value, dict) and key in ['training', 'wandb', 'advanced']:
                    flattened.update(value)
                else:
                    flattened[key] = value
            
            return flattened
        except Exception as e:
            raise ValueError(f"Error loading config file {config_path}: {e}")
    
    @classmethod
    def merge_configs(cls, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries (later configs override earlier ones)."""
        merged = {}
        for config in configs:
            if config:
                merged.update({k: v for k, v in config.items() if v is not None})
        return merged
    
    @classmethod
    def create_config(cls, 
                     cli_args: Dict[str, Any],
                     config_file: Optional[str] = None,
                     preset: Optional[str] = None) -> TrainingConfig:
        """Create a TrainingConfig from multiple sources with proper priority."""
        
        # Start with default values
        config_data = {}
        
        # Apply preset if specified
        if preset:
            preset_data = cls.load_preset(preset)
            config_data.update(preset_data)
        
        # Apply config file if specified
        if config_file:
            file_data = cls.load_config_file(config_file)
            config_data.update(file_data)
        
        # Apply CLI arguments (highest priority)
        config_data.update({k: v for k, v in cli_args.items() if v is not None})
        
        # Create and validate configuration
        return TrainingConfig(**config_data)
    
    @classmethod
    def generate_config_template(cls, preset: str = "experienced", output_path: str = "config.yaml") -> str:
        """Generate a configuration template file."""
        preset_data = cls.load_preset(preset)
        
        template = {
            "# SD3.5L LoRA Training Configuration": None,
            "dataset_url": "https://example.com/my-dataset.zip",
            "destination": "username/my-lora-model", 
            "trigger_word": "MYTOK",
            "": None,
            "# Training parameters": None,
            "training": {
                "steps": preset_data["steps"],
                "batch_size": preset_data["batch_size"],
                "optimizer": preset_data["optimizer"],
                "learning_rate": preset_data["learning_rate"],
                "resolution": preset_data["resolution"],
                "lora_rank": preset_data["lora_rank"]
            },
            " ": None,
            "# Weights & Biases integration": None,
            "wandb": {
                "project": "sd3.5_train_replicate",
                "save_interval": 100,
                "sample_interval": 100
            },
            "  ": None,
            "# Advanced settings": None,
            "advanced": {
                "caption_dropout_rate": 0.05,
                "cache_latents_to_disk": False
            }
        }
        
        # Clean up None values used for comments
        clean_template = {}
        for key, value in template.items():
            if value is not None:
                clean_template[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(clean_template, f, default_flow_style=False, sort_keys=False, indent=2)
        
        return output_path


def validate_api_token(token: Optional[str] = None) -> str:
    """Validate that Replicate API token is available and properly formatted."""
    if not token:
        token = os.getenv('REPLICATE_API_TOKEN')
    
    if not token:
        raise ValueError(
            "REPLICATE_API_TOKEN environment variable not set. "
            "Get your token from https://replicate.com/account/api-tokens"
        )
    
    if not token.startswith('r8_'):
        raise ValueError(
            "Invalid Replicate API token format. Token should start with 'r8_'"
        )
    
    return token


def validate_dataset_url(url: str) -> bool:
    """Validate that dataset URL is accessible and returns a ZIP file."""
    import requests
    from requests.exceptions import RequestException
    
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        content_length = response.headers.get('content-length')
        
        # Check if it's a ZIP file
        if 'zip' not in content_type and not url.lower().endswith('.zip'):
            raise ValueError(f"URL does not appear to be a ZIP file. Content-Type: {content_type}")
        
        # Warn about large files
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 500:
                print(f"Warning: Dataset is {size_mb:.1f}MB. Large datasets may take longer to process.")
        
        return True
        
    except RequestException as e:
        raise ValueError(f"Unable to access dataset URL: {e}")


def validate_dataset_structure(url: str) -> bool:
    """Download and validate dataset structure for proper image/caption pairing."""
    import requests
    import zipfile
    import tempfile
    import os
    from collections import defaultdict
    
    try:
        print("Downloading dataset for validation...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        try:
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                # Filter out directories and system files
                files = [f for f in file_list if not f.endswith('/') and not f.startswith('.') and not f.startswith('__MACOSX')]
                
                if not files:
                    raise ValueError("Dataset ZIP file appears to be empty or contains no valid files")
                
                # Categorize files
                image_files = []
                text_files = []
                other_files = []
                
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
                
                for file in files:
                    file_lower = file.lower()
                    base_name = os.path.splitext(file)[0]
                    ext = os.path.splitext(file_lower)[1]
                    
                    if ext in image_extensions:
                        image_files.append((base_name, file))
                    elif ext == '.txt':
                        text_files.append((base_name, file))
                    else:
                        other_files.append(file)
                
                # Check for proper pairing
                image_bases = {base for base, _ in image_files}
                text_bases = {base for base, _ in text_files}
                
                if not image_files:
                    raise ValueError("No image files found in dataset. Supported formats: " + ', '.join(image_extensions))
                
                if not text_files:
                    raise ValueError(
                        "No caption files (.txt) found in dataset.\n"
                        "IMPORTANT: Manual captioning is required - each image must have a corresponding .txt file.\n"
                        "Example: image1.jpg → image1.txt, photo2.png → photo2.txt"
                    )
                
                # Check for unpaired files
                unpaired_images = image_bases - text_bases
                unpaired_texts = text_bases - image_bases
                
                if unpaired_images:
                    raise ValueError(
                        f"Found {len(unpaired_images)} image(s) without matching caption files.\n"
                        f"Missing captions for: {', '.join(list(unpaired_images)[:5])}" + 
                        (f" and {len(unpaired_images)-5} more" if len(unpaired_images) > 5 else "") + "\n"
                        "Each image must have a corresponding .txt file with the same filename."
                    )
                
                if unpaired_texts:
                    print(f"Warning: Found {len(unpaired_texts)} caption file(s) without matching images.")
                
                # Success - report findings
                paired_count = len(image_bases & text_bases)
                print(f"✓ Dataset validation passed: {paired_count} properly paired image/caption files found")
                
                if other_files:
                    print(f"Note: {len(other_files)} other files found (will be ignored during training)")
                
                return True
                
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        if "image/caption" in str(e) or "caption" in str(e):
            raise  # Re-raise our custom validation errors
        else:
            raise ValueError(f"Failed to validate dataset structure: {e}")
