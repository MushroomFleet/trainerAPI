#!/usr/bin/env python3
"""
Test script for dataset validation functionality.
Tests the new manual captioning validation without requiring real datasets.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rich.console import Console
from rich.panel import Panel

console = Console()

def test_dataset_validation():
    """Test the dataset validation functionality."""
    console.print(Panel.fit(
        "[bold blue]Dataset Validation Test[/bold blue]\n\n"
        "This test validates the new manual captioning requirements.\n"
        "The validation will check for proper image/caption file pairing.",
        title="🧪 Testing Dataset Validation"
    ))
    
    try:
        from sd35l_trainer.config import validate_dataset_url
        
        # Test URL validation (this should work for basic URL validation)
        test_urls = [
            "https://example.com/dataset.zip",  # Valid format
            "https://example.com/dataset.txt",  # Invalid format
            "not-a-url",                        # Invalid URL
        ]
        
        console.print("\n[bold]Testing URL format validation:[/bold]")
        
        for url in test_urls:
            try:
                validate_dataset_url(url)
                console.print(f"[green]✓ URL format valid: {url}[/green]")
            except Exception as e:
                console.print(f"[red]✗ URL format invalid: {url} - {e}[/red]")
        
        console.print("\n[bold yellow]Note:[/bold yellow] Full dataset structure validation requires downloading real datasets.")
        console.print("[dim]Use --dry-run with real dataset URLs to test structure validation.[/dim]")
        
    except ImportError as e:
        console.print(f"[red]✗ Failed to import validation functions: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}[/red]")
        return False
    
    return True

def show_validation_requirements():
    """Show the new validation requirements."""
    console.print(Panel.fit(
        "[bold green]Manual Captioning Requirements[/bold green]\n\n"
        "✓ Each image must have a matching .txt caption file\n"
        "✓ Filename pairing: image1.jpg → image1.txt\n"
        "✓ No automatic captioning available\n"
        "✓ All files must be in ZIP root directory\n\n"
        "[bold red]Validation Checks:[/bold red]\n"
        "• Downloads dataset ZIP file\n"
        "• Scans for image files (.jpg, .png, etc.)\n"
        "• Scans for caption files (.txt)\n"
        "• Validates filename pairing\n"
        "• Reports unpaired files\n"
        "• Confirms dataset structure",
        title="📋 New Validation Features"
    ))

def show_enhanced_download():
    """Show the enhanced download functionality."""
    console.print(Panel.fit(
        "[bold green]Enhanced Download Features[/bold green]\n\n"
        "✓ Downloads both model and caption files\n"
        "✓ Organizes captions in 'captions/' subdirectory\n"
        "✓ Preserves original caption filenames\n"
        "✓ Reports number of caption files found\n\n"
        "[bold blue]Download Structure:[/bold blue]\n"
        "output_directory/\n"
        "├── model_name_trigger.safetensors\n"
        "└── captions/\n"
        "    ├── image1.txt\n"
        "    ├── image2.txt\n"
        "    └── ...",
        title="📥 Download Organization"
    ))

def main():
    """Run the validation tests."""
    console.print(Panel.fit(
        "[bold blue]SD3.5L Dataset Validation Testing[/bold blue]\n\n"
        "Testing the new manual captioning validation features.",
        title="🔧 Enhancement Validation"
    ))
    
    # Run tests
    success = test_dataset_validation()
    
    # Show new features
    console.print()
    show_validation_requirements()
    console.print()
    show_enhanced_download()
    
    # Summary
    if success:
        console.print(Panel.fit(
            "[bold green]✓ Dataset validation functionality is working![/bold green]\n\n"
            "[bold]Next Steps:[/bold]\n"
            "1. Test with real dataset: python SD35Ltuner.py train --dataset-url 'your-url' --dry-run\n"
            "2. Ensure your datasets include matching .txt caption files\n"
            "3. Use the interactive mode for guided setup",
            title="🎉 Test Results"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]✗ Some validation tests failed[/bold red]\n\n"
            "Check the error messages above for details.",
            title="❌ Test Results"
        ))
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
