# Python CLI Tool Development Handoff: SD3.5L LoRA Training Tool

## Project Overview

This document outlines the development requirements for a Python-based CLI tool that automates LoRA training jobs for Stable Diffusion 3.5 Large using Replicate's API. The tool will replace manual web interface interactions with a streamlined command-line experience.

## Current State Analysis

### Existing Resources
1. **JavaScript API Implementation** (`notes.txt`) - Working Replicate API integration
2. **Training Guides** - Comprehensive documentation for both beginners and experienced users
3. **Established Training Parameters** - Proven configuration values and best practices

### Target Transformation
Convert manual training workflow into automated Python CLI tool supporting remote dataset URLs and configuration management.

## Technical Requirements

### 1. Core Dependencies

```python
# Required Python packages
replicate>=0.15.0
click>=8.0.0
requests>=2.28.0
pydantic>=1.10.0
python-dotenv>=0.19.0
rich>=12.0.0  # For enhanced CLI output
pyyaml>=6.0  # For config file support
```

### 2. API Integration

#### Convert JavaScript to Python
**Current JavaScript implementation:**
```javascript
const training = await replicate.trainings.create(
  "lucataco",
  "sd3.5-large-fine-tuner", 
  "64360fd3c38f47e8132564044b67b1ed1d45b450f008b896d354c4d0d65973d0",
  {
    destination: "mushroomfleet/model-name",
    input: { /* training parameters */ }
  }
);
```

**Required Python equivalent:**
```python
import replicate

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

training = client.trainings.create(
    version="lucataco/sd3.5-large-fine-tuner:64360fd3c38f47e8132564044b67b1ed1d45b450f008b896d354c4d0d65973d0",
    input={
        "destination": destination_model,
        **training_config
    }
)
```

### 3. CLI Architecture

#### Primary Command Structure
```bash
# Basic usage
sd35l-trainer train --dataset-url "https://example.com/dataset.zip" --destination "user/model-name" --trigger-word "MYTOK"

# Advanced usage
sd35l-trainer train --config config.yaml

# Management commands
sd35l-trainer list-jobs
sd35l-trainer status <training-id>
sd35l-trainer cancel <training-id>
sd35l-trainer download <training-id>
```

#### Configuration System
Support three configuration methods (priority order):
1. Command-line arguments
2. Configuration file (YAML/JSON)
3. Default values

### 4. Training Parameters Configuration

#### Core Parameters (from training guides)
```python
class TrainingConfig:
    # Required parameters
    dataset_url: str           # Remote dataset ZIP URL
    destination: str           # "username/model-name" format
    trigger_word: str          # Unique trigger word
    
    # Optimized defaults from guides
    steps: int = 1500          # Range: 1000-2000
    lora_rank: int = 16        # Standard LoRA rank
    optimizer: str = "prodigy" # "prodigy" or "adamw8bit"
    batch_size: int = 4        # Increased from default for speed
    resolution: str = "768,1024"  # Aspect bucketing configuration
    learning_rate: float = 1.0    # For Prodigy (0.0004 for adamw8bit)
    
    # Advanced parameters
    wandb_project: str = "sd3.5_train_replicate"
    wandb_save_interval: int = 100
    caption_dropout_rate: float = 0.05
    cache_latents_to_disk: bool = False
    wandb_sample_interval: int = 100
```

#### Preset Configurations
```python
PRESETS = {
    "beginner": {
        "optimizer": "prodigy",
        "learning_rate": 1.0,
        "steps": 1500,
        "batch_size": 1,  # Conservative for beginners
        "resolution": "768,1024"
    },
    "experienced": {
        "optimizer": "prodigy", 
        "learning_rate": 1.0,
        "steps": 1500,
        "batch_size": 4,  # Faster training
        "resolution": "768,1024"
    },
    "fast": {
        "optimizer": "adamw8bit",
        "learning_rate": 0.0004,
        "steps": 1000,
        "batch_size": 4,
        "resolution": "512,768"
    }
}
```

### 5. Feature Requirements

#### Dataset URL Validation
```python
def validate_dataset_url(url: str) -> bool:
    """
    Validate that dataset URL:
    - Is accessible via HTTP HEAD request
    - Returns Content-Type indicating ZIP file
    - Has reasonable file size (warn if >500MB)
    """
    pass
```

#### Progress Monitoring
```python
def monitor_training(training_id: str) -> None:
    """
    Poll training status and display progress:
    - Training status (starting/processing/succeeded/failed)
    - Estimated completion time
    - Step progress if available
    - Error messages if failed
    """
    pass
```

#### Result Management
```python
def download_result(training_id: str, output_dir: str = "./") -> str:
    """
    Download completed training result:
    - Fetch training object
    - Download output URLs
    - Extract .safetensors file from .tar archive
    - Rename to meaningful filename: {destination}_{trigger_word}.safetensors
    """
    pass
```

### 6. CLI Command Specifications

#### Main Training Command
```bash
sd35l-trainer train [OPTIONS]

Options:
  --dataset-url TEXT          Remote ZIP dataset URL [required]
  --destination TEXT          Destination model (user/model-name) [required]
  --trigger-word TEXT         Unique trigger word [required]
  --preset [beginner|experienced|fast]  Use configuration preset
  --config FILE              Configuration file path
  --steps INTEGER            Training steps (1000-2000)
  --batch-size INTEGER       Batch size (1-4)
  --optimizer [prodigy|adamw8bit]  Optimizer choice
  --learning-rate FLOAT      Learning rate
  --resolution TEXT          Resolution setting (e.g., "768,1024")
  --wandb-project TEXT       Weights & Biases project name
  --dry-run                  Validate configuration without starting training
  --wait                     Wait for completion and download result
  --output-dir TEXT          Output directory for downloaded results
```

#### Management Commands
```bash
# List training jobs
sd35l-trainer list [--limit INT] [--status TEXT]

# Check specific training status  
sd35l-trainer status <training-id>

# Cancel training
sd35l-trainer cancel <training-id>

# Download completed training
sd35l-trainer download <training-id> [--output-dir TEXT]

# Generate configuration template
sd35l-trainer init-config [--preset TEXT] [--output FILE]
```

### 7. Configuration File Format

#### YAML Configuration Template
```yaml
# SD3.5L LoRA Training Configuration
dataset_url: "https://example.com/my-dataset.zip"
destination: "username/my-lora-model"
trigger_word: "MYTOK"

# Training parameters
training:
  steps: 1500
  batch_size: 4
  optimizer: "prodigy"
  learning_rate: 1.0
  resolution: "768,1024"
  lora_rank: 16

# Weights & Biases integration
wandb:
  project: "sd3.5_train_replicate"
  save_interval: 100
  sample_interval: 100

# Advanced settings
advanced:
  caption_dropout_rate: 0.05
  cache_latents_to_disk: false
```

### 8. Error Handling & Validation

#### Input Validation
```python
class ValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass

def validate_configuration(config: TrainingConfig) -> List[str]:
    """
    Comprehensive validation returning list of error messages:
    - API token presence and validity
    - Dataset URL accessibility
    - Destination format (username/model-name)
    - Parameter ranges (steps: 1000-2000, batch_size: 1-4, etc.)
    - Trigger word uniqueness (not common words)
    """
    pass
```

#### API Error Handling
```python
def handle_replicate_errors(func):
    """
    Decorator for Replicate API calls:
    - Handle authentication errors
    - Handle rate limiting
    - Handle network timeouts
    - Provide user-friendly error messages
    """
    pass
```

### 9. User Experience Features

#### Rich CLI Output
```python
# Use rich library for enhanced output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Display training status in formatted table
# Show progress bars for long operations
# Color-coded status messages (green=success, red=error, yellow=warning)
```

#### Interactive Mode
```python
def interactive_setup() -> TrainingConfig:
    """
    Guided setup for beginners:
    - Prompt for required parameters
    - Validate inputs in real-time
    - Suggest appropriate values
    - Explain parameter meanings
    """
    pass
```

### 10. Testing Requirements

#### Unit Tests
- Configuration validation
- Parameter conversion
- URL validation
- Error handling

#### Integration Tests  
- Replicate API connectivity
- Training job lifecycle
- File download functionality

#### End-to-End Tests
- Complete training workflow with test dataset
- Configuration file loading
- Result download and extraction

## Implementation Phases

### Phase 1: Core Functionality (Week 1)
- Basic CLI structure with click
- Replicate API integration
- Core training parameters
- Configuration validation

### Phase 2: Enhanced Features (Week 2)
- Configuration file support
- Preset management
- Progress monitoring
- Result download automation

### Phase 3: User Experience (Week 3)
- Rich CLI output
- Interactive setup mode
- Comprehensive error handling
- Documentation and examples

### Phase 4: Testing & Polish (Week 4)
- Comprehensive test suite
- Performance optimization
- Documentation finalization
- Beta user feedback integration

## Success Criteria

### Functional Requirements
- ✅ Convert manual training process to single CLI command
- ✅ Support all training parameters from existing guides
- ✅ Handle remote dataset URLs
- ✅ Provide both beginner and advanced modes
- ✅ Monitor training progress and download results

### Quality Requirements
- ✅ Comprehensive input validation
- ✅ Clear error messages with actionable suggestions
- ✅ >90% test coverage
- ✅ Sub-5 second response time for validation operations
- ✅ Graceful handling of network interruptions

### Usability Requirements
- ✅ Single command training job submission
- ✅ Intuitive parameter names matching training guides
- ✅ Configuration templates for common use cases
- ✅ Interactive mode for guided setup

## Delivery Artifacts

1. **Python Package** (`sd35l-trainer`)
   - Installable via pip
   - Entry point: `sd35l-trainer` command
   - Requirements.txt with pinned dependencies

2. **Documentation**
   - README.md with installation and usage examples
   - API documentation (if library functionality exposed)
   - Configuration reference
   - Troubleshooting guide

3. **Configuration Templates**
   - Beginner preset YAML
   - Experienced preset YAML  
   - Advanced configuration example

4. **Test Suite**
   - Unit tests for all modules
   - Integration tests for API interactions
   - Mock data for offline testing

## Technical Notes

### Environment Variables
```bash
# Required
REPLICATE_API_TOKEN=r8_your_token_here

# Optional
SD35L_DEFAULT_DESTINATION=username/default-model
SD35L_CONFIG_DIR=~/.config/sd35l-trainer
```

### File Structure
```
sd35l-trainer/
├── src/
│   └── sd35l_trainer/
│       ├── __init__.py
│       ├── cli.py              # Click CLI commands
│       ├── config.py           # Configuration management
│       ├── training.py         # Replicate API wrapper
│       ├── validation.py       # Input validation
│       ├── presets.py          # Configuration presets
│       └── utils.py           # Utility functions
├── tests/
├── docs/
├── examples/
├── setup.py
└── requirements.txt
```

This handoff document provides a comprehensive blueprint for developing a production-ready Python CLI tool that automates the SD3.5L LoRA training workflow while incorporating all the best practices and parameters from the existing training guides.
