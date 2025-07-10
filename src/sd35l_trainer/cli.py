"""
Command-line interface for SD3.5L LoRA training tool.
Provides CLI commands for training management and configuration.
"""

import sys
import os
from typing import Optional
from pathlib import Path
import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from .config import ConfigManager, TrainingConfig, validate_dataset_url
from .training import TrainingManager
from .utils import handle_errors, validate_training_id

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="sd35l-trainer")
def main():
    """SD3.5L LoRA Training CLI Tool
    
    A Python-based CLI tool for automating LoRA training jobs for 
    Stable Diffusion 3.5 Large using Replicate's API.
    
    Examples:
      sd35l-trainer train --dataset-url "https://example.com/data.zip" --destination "user/model" --trigger-word "MYTOK"
      sd35l-trainer list
      sd35l-trainer status abc123
      sd35l-trainer download abc123 --output-dir ./models
    """
    pass


@main.command()
@click.option('--dataset-url', required=True, help='Remote ZIP dataset URL')
@click.option('--destination', required=True, help='Destination model (username/model-name)')
@click.option('--trigger-word', required=True, help='Unique trigger word')
@click.option('--preset', type=click.Choice(['beginner', 'experienced', 'fast']), 
              help='Use configuration preset')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--steps', type=click.IntRange(1000, 2000), help='Training steps (1000-2000)')
@click.option('--batch-size', type=click.IntRange(1, 8), help='Batch size (1-8)')
@click.option('--optimizer', type=click.Choice(['prodigy', 'adamw8bit']), help='Optimizer choice')
@click.option('--learning-rate', type=float, help='Learning rate')
@click.option('--resolution', help='Resolution setting (e.g., "768,1024")')
@click.option('--lora-rank', type=click.IntRange(4, 128), help='LoRA rank (4-128)')
@click.option('--wandb-project', help='Weights & Biases project name')
@click.option('--dry-run', is_flag=True, help='Validate configuration without starting training')
@click.option('--wait', is_flag=True, help='Wait for completion and download result')
@click.option('--output-dir', default='./', help='Output directory for downloaded results')
@handle_errors
def train(dataset_url: str, destination: str, trigger_word: str, preset: Optional[str],
          config: Optional[str], steps: Optional[int], batch_size: Optional[int],
          optimizer: Optional[str], learning_rate: Optional[float], resolution: Optional[str],
          lora_rank: Optional[int], wandb_project: Optional[str], dry_run: bool,
          wait: bool, output_dir: str):
    """Start a new LoRA training job.
    
    This command submits a training job to Replicate using the specified parameters.
    Required parameters are dataset URL, destination model name, and trigger word.
    
    Examples:
      # Basic training with preset
      sd35l-trainer train --dataset-url "https://example.com/data.zip" \\
                          --destination "user/my-model" \\
                          --trigger-word "MYTOK" \\
                          --preset experienced
    
      # Advanced training with custom parameters
      sd35l-trainer train --dataset-url "https://example.com/data.zip" \\
                          --destination "user/my-model" \\
                          --trigger-word "MYTOK" \\
                          --steps 1500 \\
                          --batch-size 4 \\
                          --optimizer prodigy \\
                          --learning-rate 1.0
    """
    
    # Validate dataset URL and structure
    console.print("[yellow]Validating dataset URL and structure...[/yellow]")
    try:
        from .config import validate_dataset_structure
        validate_dataset_url(dataset_url)
        console.print("[green]✓ Dataset URL is accessible[/green]")
        
        # Enhanced validation for image/caption pairing
        validate_dataset_structure(dataset_url)
        console.print("[green]✓ Dataset structure validated - proper image/caption pairing found[/green]")
    except ValueError as e:
        console.print(f"[red]✗ Dataset validation failed: {e}[/red]")
        sys.exit(1)
    
    # Gather CLI arguments
    cli_args = {
        'dataset_url': dataset_url,
        'destination': destination,
        'trigger_word': trigger_word,
        'steps': steps,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'resolution': resolution,
        'lora_rank': lora_rank,
        'wandb_project': wandb_project
    }
    
    # Create configuration
    try:
        training_config = ConfigManager.create_config(
            cli_args=cli_args,
            config_file=config,
            preset=preset
        )
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {e}[/red]")
        sys.exit(1)
    
    # Initialize training manager
    try:
        trainer = TrainingManager()
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize training manager: {e}[/red]")
        sys.exit(1)
    
    # Submit training
    try:
        training_id = trainer.submit_training(training_config, dry_run=dry_run)
        
        if dry_run:
            console.print("[green]Configuration validation completed successfully![/green]")
            return
        
        if not training_id:
            console.print("[red]✗ Failed to get training ID[/red]")
            sys.exit(1)
        
        # Wait for completion if requested
        if wait:
            console.print("\n[yellow]Waiting for training to complete...[/yellow]")
            final_status = trainer.monitor_training(training_id)
            
            if final_status == 'succeeded':
                console.print("\n[yellow]Downloading training result...[/yellow]")
                result_path = trainer.download_result(training_id, output_dir)
                if result_path:
                    console.print(f"\n[green]✓ Training completed and result downloaded to: {result_path}[/green]")
                else:
                    console.print("\n[yellow]Training completed but download failed[/yellow]")
            elif final_status in ['failed', 'canceled']:
                console.print(f"\n[red]Training ended with status: {final_status}[/red]")
                sys.exit(1)
        else:
            console.print(f"\n[green]Training job submitted! Use 'sd35l-trainer status {training_id}' to check progress.[/green]")
            
    except Exception as e:
        console.print(f"[red]✗ Training submission failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--limit', default=10, help='Number of results to show (default: 10)')
@click.option('--status', help='Filter by status (starting, processing, succeeded, failed, canceled)')
@handle_errors
def list(limit: int, status: Optional[str]):
    """List recent training jobs.
    
    Shows a table of recent training jobs with their status, destination, and creation time.
    
    Examples:
      sd35l-trainer list
      sd35l-trainer list --limit 20
      sd35l-trainer list --status processing
    """
    
    try:
        trainer = TrainingManager()
        trainings = trainer.list_trainings(limit=limit)
        
        # Filter by status if specified
        if status:
            trainings = [t for t in trainings if t['status'] == status]
        
        trainer.display_training_list(trainings)
        
    except Exception as e:
        console.print(f"[red]✗ Failed to list trainings: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('training_id')
@handle_errors
def status(training_id: str):
    """Check the status of a specific training job.
    
    Displays detailed information about a training job including current status,
    configuration parameters, and timing information.
    
    Example:
      sd35l-trainer status abc123def456
    """
    
    validate_training_id(training_id)
    
    try:
        trainer = TrainingManager()
        trainer.display_training_status(training_id)
        
    except Exception as e:
        console.print(f"[red]✗ Failed to get training status: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('training_id')
@handle_errors
def cancel(training_id: str):
    """Cancel a running training job.
    
    Attempts to cancel a training job. Only jobs in 'starting' or 'processing' 
    status can be canceled.
    
    Example:
      sd35l-trainer cancel abc123def456
    """
    
    validate_training_id(training_id)
    
    # Confirm cancellation
    if not Confirm.ask(f"[yellow]Are you sure you want to cancel training {training_id}?[/yellow]"):
        console.print("[dim]Cancellation aborted[/dim]")
        return
    
    try:
        trainer = TrainingManager()
        success = trainer.cancel_training(training_id)
        
        if success:
            console.print(f"[green]✓ Cancellation requested for training {training_id}[/green]")
        else:
            console.print(f"[yellow]Training {training_id} could not be canceled[/yellow]")
            
    except Exception as e:
        console.print(f"[red]✗ Failed to cancel training: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('training_id')
@click.option('--output-dir', default='./', help='Output directory for downloaded files')
@handle_errors
def download(training_id: str, output_dir: str):
    """Download the result of a completed training job.
    
    Downloads and extracts the .safetensors file from a completed training job.
    The file will be renamed to include the model name and trigger word.
    
    Example:
      sd35l-trainer download abc123def456 --output-dir ./models
    """
    
    validate_training_id(training_id)
    
    try:
        trainer = TrainingManager()
        result_path = trainer.download_result(training_id, output_dir)
        
        if result_path:
            console.print(f"[green]✓ Training result downloaded to: {result_path}[/green]")
        else:
            console.print("[red]✗ Failed to download training result[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--preset', default='experienced', type=click.Choice(['beginner', 'experienced', 'fast']),
              help='Configuration preset to use as base (default: experienced)')
@click.option('--output', default='config.yaml', help='Output file path (default: config.yaml)')
@handle_errors
def init_config(preset: str, output: str):
    """Generate a configuration template file.
    
    Creates a YAML configuration file with all available parameters and their
    descriptions. Use this as a starting point for custom configurations.
    
    Examples:
      sd35l-trainer init-config
      sd35l-trainer init-config --preset beginner --output my-config.yaml
    """
    
    try:
        output_path = ConfigManager.generate_config_template(preset=preset, output_path=output)
        console.print(f"[green]✓ Configuration template created: {output_path}[/green]")
        console.print(f"[dim]Edit the file and use with: sd35l-trainer train --config {output_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to generate config template: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--token', help='Replicate API token (optional, will prompt if not provided)')
@handle_errors
def setup(token: Optional[str]):
    """Set up the .env file with your Replicate API token.
    
    Creates a .env file in the current directory with your API token.
    This is an alternative to setting the REPLICATE_API_TOKEN environment variable.
    
    Example:
      sd35l-trainer setup
      sd35l-trainer setup --token r8_your_token_here
    """
    
    env_file_path = Path('.env')
    
    # Check if .env already exists
    if env_file_path.exists():
        if not Confirm.ask(f"[yellow].env file already exists. Overwrite?[/yellow]"):
            console.print("[dim]Setup canceled[/dim]")
            return
    
    # Get token if not provided
    if not token:
        console.print(Panel.fit(
            "[bold blue]Replicate API Token Setup[/bold blue]\n\n"
            "To use this tool, you need a Replicate API token.\n\n"
            "1. Go to: https://replicate.com/account/api-tokens\n"
            "2. Create a new token or copy an existing one\n"
            "3. Paste it below",
            title="Setup Required"
        ))
        
        # Manual validation loop for compatibility with Rich 12.x
        while True:
            token = Prompt.ask(
                "\n[bold]Enter your Replicate API token[/bold]",
                password=True
            )
            
            if token and token.startswith('r8_'):
                break
            else:
                console.print("[red]✗ Invalid token format. Token should start with 'r8_'. Please try again.[/red]")
    
    # Validate token format
    if not token or not token.startswith('r8_'):
        console.print("[red]✗ Invalid token format. Token should start with 'r8_'[/red]")
        return
    
    # Create .env file
    try:
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# SD3.5L LoRA Training CLI Tool Environment Variables\n")
            f.write(f"# Generated by sd35l-trainer setup command\n\n")
            f.write(f"REPLICATE_API_TOKEN={token}\n")
            f.write(f"\n# Optional: Default destination for models\n")
            f.write(f"# SD35L_DEFAULT_DESTINATION=yourusername/default-model\n")
        
        console.print(f"[green]✓ .env file created successfully![/green]")
        console.print(f"[dim]Your API token has been saved to .env[/dim]")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"[dim]1. Try: python SD35Ltuner.py interactive[/dim]")
        console.print(f"[dim]2. Or: python SD35Ltuner.py train --help[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to create .env file: {e}[/red]")


@main.command()
@handle_errors
def interactive():
    """Interactive setup mode for beginners.
    
    Guides you through the configuration process with prompts and explanations.
    Ideal for users new to LoRA training who want step-by-step assistance.
    """
    
    console.print(Panel.fit(
        "[bold blue]SD3.5L LoRA Training - Interactive Setup[/bold blue]\n\n"
        "This wizard will guide you through setting up a LoRA training job.\n"
        "You can press Ctrl+C at any time to cancel.",
        title="Welcome"
    ))
    
    try:
        # Required parameters - manual validation for Rich 12.x compatibility
        while True:
            dataset_url = Prompt.ask(
                "\n[bold]Dataset URL[/bold]\n"
                "Enter the URL of your ZIP file containing training images"
            )
            try:
                validate_dataset_url(dataset_url)
                break
            except ValueError as e:
                console.print(f"[red]✗ {e}[/red]")
                console.print("[yellow]Please enter a valid dataset URL.[/yellow]")
        
        while True:
            destination = Prompt.ask(
                "\n[bold]Destination Model[/bold]\n"
                "Enter destination in format 'username/model-name'"
            )
            if '/' in destination and len(destination.split('/')) == 2:
                break
            else:
                console.print("[red]✗ Invalid format. Please use 'username/model-name'.[/red]")
        
        trigger_word = Prompt.ask(
            "\n[bold]Trigger Word[/bold]\n"
            "Enter a unique trigger word (avoid common words like 'person', 'style')"
        )
        
        # Preset selection
        preset = Prompt.ask(
            "\n[bold]Training Preset[/bold]\n"
            "Choose a preset configuration",
            choices=['beginner', 'experienced', 'fast'],
            default='experienced'
        )
        
        # Confirmation
        config_preview = f"""
Dataset URL: {dataset_url}
Destination: {destination}  
Trigger Word: {trigger_word}
Preset: {preset}
"""
        
        console.print(Panel.fit(config_preview, title="Configuration Preview"))
        
        if Confirm.ask("\n[bold]Start training with this configuration?[/bold]"):
            # Start training
            ctx = click.get_current_context()
            ctx.invoke(train, 
                      dataset_url=dataset_url,
                      destination=destination, 
                      trigger_word=trigger_word,
                      preset=preset,
                      config=None,
                      steps=None,
                      batch_size=None,
                      optimizer=None,
                      learning_rate=None,
                      resolution=None,
                      lora_rank=None,
                      wandb_project=None,
                      dry_run=False,
                      wait=False,
                      output_dir='./')
        else:
            console.print("[dim]Training canceled[/dim]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive setup canceled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Interactive setup failed: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()
