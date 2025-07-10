# LoRA Training Guide for Stable Diffusion 3.5 Large - Experienced Users

## Overview

This guide covers LoRA (Low-Rank Adaptation) training for Stable Diffusion 3.5 Large, assuming familiarity with machine learning concepts and model training workflows.

## Prerequisites

* Understanding of diffusion models and fine-tuning concepts
* Familiarity with training hyperparameters
* Access to SD 3.5L training platform
* Dataset of 10-50 high-quality images (minimum)

## Training Process

### 1\. Image Collection

* Curate high-quality, consistent dataset (1024px+ resolution recommended)
* Maintain visual coherence and style consistency
* Remove low-quality, blurry, or heavily compressed images

### 2\. Captioning (Optional)

* Manual captioning provides better control over training signals
* Use descriptive, consistent terminology
* Focus on key visual elements and style characteristics
* Skip if using auto-captioning in step 4

### 3\. Dataset Preparation

* Archive images in .zip format
* Maintain flat directory structure (no subdirectories)
* Ensure file naming consistency
* Upload to training platform

### 4\. Auto-Captioning Configuration

* Enable auto-captioning if manual captions not provided
* Review generated captions for accuracy
* Platform will analyze visual content and generate descriptive text

### 5\. Trigger Word Selection

* Choose unique, memorable trigger word
* Avoid common dictionary words to prevent conflicts
* Consider using person names, invented terms, or specific identifiers
* Will be automatically prepended to training captions

### 6\. Optimizer Configuration

* **Optimizer**: Prodigy (adaptive learning rate optimizer)
* **Learning Rate**: 1.0 (Prodigy handles adaptive scaling)
* Prodigy eliminates need for LR scheduling and manual tuning

### 7\. Batch Size Optimization

* **Batch Size**: 4 (increased from default)
* Improves training stability and convergence speed
* Balance between memory usage and training efficiency

### 8\. Training Duration

* **Steps**: 1000-2000 (adjust based on dataset size and convergence)
* Monitor training loss curves if available
* Larger datasets may require more steps

### 9\. Resolution Configuration

* **Setting**: "768,1024" (aspect bucketing parameters)
* Enables multi-aspect ratio training through aspect bucketing
* Allows model to learn from various image orientations efficiently
* Superior to fixed-resolution training for diverse datasets

### 10\. Training Execution

* Initiate training with configured parameters
* Monitor progress through platform interface
* Training time varies based on dataset size and hardware

## Post-Training

### Model Retrieval

* Download .tar archive containing trained LoRA
* Extract .safetensors file
* Rename to descriptive filename for organization

### Integration

* Load LoRA with base SD 3.5L model
* Test with trigger word in prompts
* Adjust LoRA strength (0.7-1.0 typically optimal)

## Optimization Tips

* **Dataset Quality > Quantity**: 20 high-quality images often outperform 100 mediocre ones
* **Consistent Captioning**: Maintain vocabulary consistency across captions
* **Aspect Bucketing**: The 768,1024 setting automatically handles diverse aspect ratios
* **Convergence Monitoring**: Stop training if loss plateaus early to prevent overfitting

## Technical Notes

* Prodigy optimizer adapts learning rate automatically based on training dynamics
* Aspect bucketing reduces the need for image preprocessing and cropping
* Higher batch sizes improve gradient stability but increase memory requirements
* LoRA rank and alpha are typically handled automatically by the platform
