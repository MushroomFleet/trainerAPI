# SD3.5L LoRA Training CLI Tool

A Python-based CLI tool for automating LoRA training jobs for Stable Diffusion 3.5 Large using Replicate's API. This tool converts the manual web interface workflow into a streamlined command-line experience.

## Features

- **Easy Training Setup**: Single command to start LoRA training with optimal parameters
- **Multiple Configuration Options**: CLI arguments, configuration files, and presets
- **Interactive Mode**: Guided setup for beginners with step-by-step prompts
- **Training Management**: List, monitor, cancel, and download training jobs
- **Rich CLI Output**: Beautiful terminal interface with progress bars and status tables
- **Preset Configurations**: Beginner, experienced, and fast training presets
- **Automatic Result Management**: Downloads and renames .safetensors files automatically

## Installation

### Prerequisites

- Python 3.7 or higher
- Replicate API account and token

### Setup

1. **Clone or download this repository**
   ```bash
   cd sd35L-trainer/trainerAPI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Replicate API token (choose one method):**

   **Option A: Using .env file (recommended):**
   ```bash
   python SD35Ltuner.py setup
   ```
   
   **Option B: Environment variable:**
   ```bash
   export REPLICATE_API_TOKEN=r8_your_token_here
   ```
   
   **Option C: Manual .env file:**
   ```bash
   echo "REPLICATE_API_TOKEN=r8_your_token_here" > .env
   ```
   
   Get your token from: https://replicate.com/account/api-tokens

4. **Make the tool executable**
   ```bash
   # On Windows
   python SD35Ltuner.py --help
   
   # On Linux/Mac
   chmod +x SD35Ltuner.py
   ./SD35Ltuner.py --help
   ```

## Quick Start

### Basic Training

```bash
python SD35Ltuner.py train \
  --dataset-url "https://example.com/my-dataset.zip" \
  --destination "username/my-lora-model" \
  --trigger-word "MYTOK" \
  --preset experienced
```

### Interactive Setup (Recommended for Beginners)

```bash
python SD35Ltuner.py interactive
```

### Advanced Training with Custom Parameters

```bash
python SD35Ltuner.py train \
  --dataset-url "https://example.com/dataset.zip" \
  --destination "username/my-model" \
  --trigger-word "ZEPHIRA" \
  --steps 1500 \
  --batch-size 4 \
  --optimizer prodigy \
  --learning-rate 1.0 \
  --resolution "768,1024" \
  --wait
```

## Configuration Presets

### Beginner
- **Optimizer**: Prodigy (adaptive learning rate)
- **Learning Rate**: 1.0
- **Steps**: 1500
- **Batch Size**: 1 (conservative for stability)
- **Resolution**: 768,1024

### Experienced
- **Optimizer**: Prodigy
- **Learning Rate**: 1.0
- **Steps**: 1500
- **Batch Size**: 4 (faster training)
- **Resolution**: 768,1024

### Fast
- **Optimizer**: adamw8bit
- **Learning Rate**: 0.0004
- **Steps**: 1000
- **Batch Size**: 4
- **Resolution**: 512,768

## Commands

### Training Commands

#### `train` - Start a new training job
```bash
python SD35Ltuner.py train [OPTIONS]

Required Options:
  --dataset-url TEXT          Remote ZIP dataset URL
  --destination TEXT          Destination model (username/model-name)
  --trigger-word TEXT         Unique trigger word

Optional Parameters:
  --preset [beginner|experienced|fast]  Use configuration preset
  --config FILE                         Configuration file path
  --steps INTEGER                       Training steps (1000-2000)
  --batch-size INTEGER                  Batch size (1-8)
  --optimizer [prodigy|adamw8bit]       Optimizer choice
  --learning-rate FLOAT                 Learning rate
  --resolution TEXT                     Resolution (e.g., "768,1024")
  --lora-rank INTEGER                   LoRA rank (4-128)
  --wandb-project TEXT                  W&B project name
  --dry-run                             Validate config without training
  --wait                                Wait for completion and download
  --output-dir TEXT                     Download directory
```

#### `interactive` - Guided setup
```bash
python SD35Ltuner.py interactive
```

### Management Commands

#### `list` - List training jobs
```bash
python SD35Ltuner.py list [--limit 10] [--status processing]
```

#### `status` - Check training status
```bash
python SD35Ltuner.py status <training-id>
```

#### `cancel` - Cancel training
```bash
python SD35Ltuner.py cancel <training-id>
```

#### `download` - Download results
```bash
python SD35Ltuner.py download <training-id> [--output-dir ./models]
```

### Configuration Commands

#### `init-config` - Generate config template
```bash
python SD35Ltuner.py init-config [--preset experienced] [--output config.yaml]
```

## Configuration Files

### Creating a Configuration File

Generate a template:
```bash
python SD35Ltuner.py init-config --preset experienced
```

### YAML Configuration Example

```yaml
# SD3.5L LoRA Training Configuration
dataset_url: "https://example.com/my-dataset.zip"
destination: "username/my-lora-model"
trigger_word: "MYTOK"

training:
  steps: 1500
  batch_size: 4
  optimizer: "prodigy"
  learning_rate: 1.0
  resolution: "768,1024"
  lora_rank: 16

wandb:
  project: "sd3.5_train_replicate"
  save_interval: 100
  sample_interval: 100

advanced:
  caption_dropout_rate: 0.05
  cache_latents_to_disk: false
```

### Using Configuration Files

```bash
python SD35Ltuner.py train --config my-config.yaml
```

## Dataset Requirements

### Image Requirements
- **Format**: ZIP file containing images AND caption files
- **Image Count**: 10-50 high-quality images (15-30 recommended)
- **Image Size**: Minimum 512x512 pixels
- **Image Quality**: Clear, well-lit, non-blurry images
- **Consistency**: Similar style, lighting, or subject across images

### Caption Requirements ⚠️ CRITICAL
- **Manual Captioning Required**: Each image MUST have a corresponding .txt file
- **Filename Matching**: Caption files must have the same name as image files
- **No Auto-Captioning**: This API does not provide automatic captioning
- **Caption Content**: Describe what you want the model to learn about each image

### Dataset URL
- Must be publicly accessible HTTP/HTTPS URL
- Must return a ZIP file (Content-Type or .zip extension)
- Recommended: Under 500MB for faster processing

### File Structure
```
dataset.zip
├── image1.jpg
├── image1.txt      ← Caption for image1.jpg
├── image2.png
├── image2.txt      ← Caption for image2.png
├── image3.jpg
├── image3.txt      ← Caption for image3.jpg
└── ...
```
**CRITICAL**: Each image must have a matching .txt caption file. Place all files directly in the ZIP root, not in subdirectories.

## Training Parameters Guide

### Essential Parameters

- **Dataset URL**: Public URL to your ZIP file containing training images
- **Destination**: Where to save the trained model (format: `username/model-name`)
- **Trigger Word**: Unique word to activate your LoRA (avoid common words)

### Optimizer Settings

**Prodigy (Recommended)**
- Adaptive learning rate optimizer
- Learning rate: 1.0 (automatic adjustment)
- Best for most use cases

**AdamW8bit**
- Traditional optimizer with manual learning rate
- Learning rate: 0.0004 (requires manual tuning)
- Faster but requires more expertise

### Training Duration

- **1000 steps**: Good for smaller datasets or faster training
- **1500 steps**: Recommended for most use cases
- **2000 steps**: For larger datasets or fine-tuning

### Batch Size

- **1**: Conservative, uses less memory, slower training
- **4**: Recommended balance of speed and stability
- **8**: Fastest but requires more memory

### Resolution

- **512,768**: Faster training, smaller images
- **768,1024**: Recommended for quality (aspect bucketing)
- **1024,1024**: Highest quality, slower training

## Examples

### Example 1: Portrait LoRA
```bash
python SD35Ltuner.py train \
  --dataset-url "https://myserver.com/portraits.zip" \
  --destination "myuser/portrait-style" \
  --trigger-word "PORTRAITSTYLE" \
  --preset experienced \
  --wait \
  --output-dir ./completed-models
```

### Example 2: Character LoRA with Custom Settings
```bash
python SD35Ltuner.py train \
  --dataset-url "https://storage.googleapis.com/my-character-data.zip" \
  --destination "artist/fantasy-character" \
  --trigger-word "ZEPHIRA" \
  --steps 1800 \
  --batch-size 4 \
  --optimizer prodigy \
  --learning-rate 1.0 \
  --resolution "768,1024"
```

### Example 3: Fast Style Training
```bash
python SD35Ltuner.py train \
  --dataset-url "https://example.com/art-style.zip" \
  --destination "studio/art-style-v1" \
  --trigger-word "ARTSTYLE" \
  --preset fast
```

### Example 4: Using Configuration File
```bash
# Generate template
python SD35Ltuner.py init-config --preset beginner --output my-training.yaml

# Edit my-training.yaml with your parameters

# Start training
python SD35Ltuner.py train --config my-training.yaml --wait
```

## Troubleshooting

### Common Issues

**"REPLICATE_API_TOKEN environment variable not set"**
- Set your API token: `export REPLICATE_API_TOKEN=r8_your_token_here`
- Get token from: https://replicate.com/account/api-tokens

**"Dataset URL validation failed"**
- Ensure URL is publicly accessible
- Verify it returns a ZIP file
- Check file size (recommended <500MB)

**"Trigger word is too common"**
- Use unique words like "ZEPHIRA", "MYSTYLE", or your name
- Avoid common words like "person", "style", "art"

**"Destination must be in format 'username/model-name'"**
- Use format: `yourusername/your-model-name`
- Create the model on Replicate first if needed

### Getting Help

1. **Check command help**: `python SD35Ltuner.py train --help`
2. **Use dry-run mode**: `--dry-run` to validate configuration
3. **Start with presets**: Use `--preset experienced` for proven settings
4. **Try interactive mode**: `python SD35Ltuner.py interactive`

### Performance Tips

- Use batch size 4 for optimal speed/stability balance
- Keep datasets under 500MB for faster processing
- Use Prodigy optimizer with learning rate 1.0
- Monitor training with `status` command
- Use `--wait` flag for automatic download

## Advanced Usage

### Monitoring Training Progress

```bash
# Submit training
TRAINING_ID=$(python SD35Ltuner.py train --dataset-url "..." --destination "..." --trigger-word "..." | grep "Training ID:" | cut -d' ' -f3)

# Monitor status
python SD35Ltuner.py status $TRAINING_ID

# Download when complete
python SD35Ltuner.py download $TRAINING_ID --output-dir ./models
```

### Batch Processing Multiple Datasets

```bash
# Create multiple config files
python SD35Ltuner.py init-config --preset experienced --output config1.yaml
python SD35Ltuner.py init-config --preset experienced --output config2.yaml

# Edit configs with different datasets

# Submit multiple trainings
python SD35Ltuner.py train --config config1.yaml
python SD35Ltuner.py train --config config2.yaml

# List all jobs
python SD35Ltuner.py list --limit 20
```

## Contributing

This tool is based on the comprehensive specification in `PYDEV-HANDOFF.md` and incorporates best practices from the included training guides.

## License

This project is provided as-is for educational and research purposes.
