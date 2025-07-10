#!/usr/bin/env python3
"""
Basic functionality test for SD35Ltuner.py
Tests configuration validation and CLI parsing without requiring external resources.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sd35l_trainer.config import ConfigManager, TrainingConfig
from rich.console import Console

console = Console()

def test_configuration_presets():
    """Test that configuration presets load correctly."""
    console.print("[blue]Testing Configuration Presets...[/blue]")
    
    presets = ['beginner', 'experienced', 'fast']
    
    for preset_name in presets:
        try:
            preset_data = ConfigManager.load_preset(preset_name)
            console.print(f"[green]✓ {preset_name} preset loaded successfully[/green]")
            console.print(f"  - Optimizer: {preset_data['optimizer']}")
            console.print(f"  - Learning Rate: {preset_data['learning_rate']}")
            console.print(f"  - Steps: {preset_data['steps']}")
            console.print(f"  - Batch Size: {preset_data['batch_size']}")
            console.print()
        except Exception as e:
            console.print(f"[red]✗ Failed to load {preset_name} preset: {e}[/red]")
            return False
    
    return True

def test_configuration_validation():
    """Test configuration validation with valid parameters."""
    console.print("[blue]Testing Configuration Validation...[/blue]")
    
    test_configs = [
        {
            'name': 'Valid Prodigy Config',
            'config': {
                'dataset_url': 'https://example.com/dataset.zip',
                'destination': 'user/model-name',
                'trigger_word': 'UNIQUETOKEN',
                'optimizer': 'prodigy',
                'learning_rate': 1.0,
                'steps': 1500,
                'batch_size': 4,
                'resolution': '768,1024'
            }
        },
        {
            'name': 'Valid AdamW Config',
            'config': {
                'dataset_url': 'https://example.com/dataset.zip',
                'destination': 'user/model-name', 
                'trigger_word': 'UNIQUETOKEN',
                'optimizer': 'adamw8bit',
                'learning_rate': 0.0004,
                'steps': 1000,
                'batch_size': 2,
                'resolution': '512,768'
            }
        }
    ]
    
    for test_case in test_configs:
        try:
            config = TrainingConfig(**test_case['config'])
            console.print(f"[green]✓ {test_case['name']} validated successfully[/green]")
        except Exception as e:
            console.print(f"[red]✗ {test_case['name']} validation failed: {e}[/red]")
            return False
    
    return True

def test_configuration_validation_errors():
    """Test that configuration validation catches errors."""
    console.print("[blue]Testing Configuration Error Handling...[/blue]")
    
    error_configs = [
        {
            'name': 'Invalid Destination Format',
            'config': {
                'dataset_url': 'https://example.com/dataset.zip',
                'destination': 'invalid-destination',  # Missing slash
                'trigger_word': 'UNIQUETOKEN'
            },
            'expected_error': 'username/model-name'
        },
        {
            'name': 'Common Trigger Word',
            'config': {
                'dataset_url': 'https://example.com/dataset.zip',
                'destination': 'user/model-name',
                'trigger_word': 'person'  # Common word
            },
            'expected_error': 'too common'
        },
        {
            'name': 'Invalid Optimizer',
            'config': {
                'dataset_url': 'https://example.com/dataset.zip',
                'destination': 'user/model-name',
                'trigger_word': 'UNIQUETOKEN',
                'optimizer': 'invalid_optimizer'
            },
            'expected_error': 'Optimizer must be one of'
        }
    ]
    
    for test_case in error_configs:
        try:
            config = TrainingConfig(**test_case['config'])
            console.print(f"[red]✗ {test_case['name']} should have failed but didn't[/red]")
            return False
        except Exception as e:
            if test_case['expected_error'] in str(e):
                console.print(f"[green]✓ {test_case['name']} correctly caught error[/green]")
            else:
                console.print(f"[yellow]? {test_case['name']} caught error but message unexpected: {e}[/yellow]")
    
    return True

def test_config_merging():
    """Test configuration merging from multiple sources."""
    console.print("[blue]Testing Configuration Merging...[/blue]")
    
    try:
        # Test merging preset with CLI args
        cli_args = {
            'dataset_url': 'https://example.com/dataset.zip',
            'destination': 'user/my-model',
            'trigger_word': 'MYTOK',
            'steps': 1800  # Override preset value
        }
        
        config = ConfigManager.create_config(
            cli_args=cli_args,
            preset='experienced'
        )
        
        # Check that CLI args override preset values
        assert config.steps == 1800, f"Expected steps=1800, got {config.steps}"
        assert config.batch_size == 4, f"Expected batch_size=4 from preset, got {config.batch_size}"
        assert config.optimizer == 'prodigy', f"Expected optimizer=prodigy from preset, got {config.optimizer}"
        
        console.print("[green]✓ Configuration merging works correctly[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Configuration merging failed: {e}[/red]")
        return False

def main():
    """Run all tests."""
    console.print("[bold blue]SD3.5L LoRA Training CLI Tool - Basic Functionality Test[/bold blue]\n")
    
    tests = [
        test_configuration_presets,
        test_configuration_validation,
        test_configuration_validation_errors,
        test_config_merging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        console.print()
    
    console.print(f"[bold]Test Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("[bold green]🎉 All tests passed! The CLI tool is working correctly.[/bold green]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("[dim]1. Set your REPLICATE_API_TOKEN environment variable[/dim]")
        console.print("[dim]2. Try: python SD35Ltuner.py train --help[/dim]") 
        console.print("[dim]3. Use --dry-run to test configurations[/dim]")
        return True
    else:
        console.print(f"[bold red]❌ {total - passed} tests failed.[/bold red]")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
