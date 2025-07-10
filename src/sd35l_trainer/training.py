"""
Replicate API integration for SD3.5L LoRA training.
Handles training job submission, monitoring, and result management.
"""

import os
import time
import tarfile
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path
import replicate
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .config import TrainingConfig, validate_api_token

console = Console()


class TrainingManager:
    """Manages Replicate training jobs for SD3.5L LoRA training."""
    
    # Replicate model version for SD3.5L fine-tuner
    MODEL_VERSION = "lucataco/sd3.5-large-fine-tuner:64360fd3c38f47e8132564044b67b1ed1d45b450f008b896d354c4d0d65973d0"
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize training manager with Replicate client."""
        self.api_token = validate_api_token(api_token)
        self.client = replicate.Client(api_token=self.api_token)
    
    def submit_training(self, config: TrainingConfig, dry_run: bool = False) -> Optional[str]:
        """Submit a training job to Replicate."""
        
        # Prepare training input parameters
        training_input = {
            "steps": config.steps,
            "lora_rank": config.lora_rank,
            "optimizer": config.optimizer,
            "batch_size": config.batch_size,
            "resolution": config.resolution,
            "input_images": config.dataset_url,
            "trigger_word": config.trigger_word,
            "learning_rate": config.learning_rate,
            "wandb_project": config.wandb_project,
            "wandb_save_interval": config.wandb_save_interval,
            "caption_dropout_rate": config.caption_dropout_rate,
            "cache_latents_to_disk": config.cache_latents_to_disk,
            "wandb_sample_interval": config.wandb_sample_interval
        }
        
        if dry_run:
            console.print(Panel.fit(
                f"[bold green]Dry Run - Training Configuration Validated[/bold green]\n\n"
                f"[bold]Destination:[/bold] {config.destination}\n"
                f"[bold]Trigger Word:[/bold] {config.trigger_word}\n"
                f"[bold]Dataset URL:[/bold] {config.dataset_url}\n"
                f"[bold]Steps:[/bold] {config.steps}\n"
                f"[bold]Batch Size:[/bold] {config.batch_size}\n"
                f"[bold]Optimizer:[/bold] {config.optimizer}\n"
                f"[bold]Learning Rate:[/bold] {config.learning_rate}\n"
                f"[bold]Resolution:[/bold] {config.resolution}",
                title="Training Configuration"
            ))
            return None
        
        try:
            console.print(f"[yellow]Submitting training job to Replicate...[/yellow]")
            
            training = self.client.trainings.create(
                version=self.MODEL_VERSION,
                input=training_input,
                destination=config.destination
            )
            
            training_id = training.id
            console.print(f"[green]✓ Training job submitted successfully![/green]")
            console.print(f"[bold]Training ID:[/bold] {training_id}")
            console.print(f"[bold]Status URL:[/bold] https://replicate.com/p/{training_id}")
            
            return training_id
            
        except Exception as e:
            console.print(f"[red]✗ Failed to submit training job: {e}[/red]")
            raise
    
    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        try:
            training = self.client.trainings.get(training_id)
            
            return {
                "id": training.id,
                "status": training.status,
                "created_at": training.created_at,
                "started_at": getattr(training, 'started_at', None),
                "completed_at": getattr(training, 'completed_at', None),
                "source": training.source,
                "destination": training.destination,
                "input": training.input,
                "output": getattr(training, 'output', None),
                "error": getattr(training, 'error', None),
                "logs": getattr(training, 'logs', None)
            }
        except Exception as e:
            console.print(f"[red]✗ Failed to get training status: {e}[/red]")
            raise
    
    def list_trainings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent training jobs."""
        try:
            trainings = list(self.client.trainings.list())[:limit]
            return [
                {
                    "id": t.id,
                    "status": t.status,
                    "created_at": t.created_at,
                    "destination": t.destination,
                    "source": t.source
                }
                for t in trainings
            ]
        except Exception as e:
            console.print(f"[red]✗ Failed to list trainings: {e}[/red]")
            raise
    
    def cancel_training(self, training_id: str) -> bool:
        """Cancel a training job."""
        try:
            training = self.client.trainings.get(training_id)
            if training.status in ['starting', 'processing']:
                training.cancel()
                console.print(f"[yellow]Training {training_id} cancellation requested[/yellow]")
                return True
            else:
                console.print(f"[yellow]Training {training_id} cannot be cancelled (status: {training.status})[/yellow]")
                return False
        except Exception as e:
            console.print(f"[red]✗ Failed to cancel training: {e}[/red]")
            raise
    
    def monitor_training(self, training_id: str, check_interval: int = 30) -> str:
        """Monitor a training job until completion."""
        console.print(f"[yellow]Monitoring training {training_id}...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training in progress...", total=None)
            
            while True:
                try:
                    status_info = self.get_training_status(training_id)
                    status = status_info['status']
                    
                    if status == 'succeeded':
                        progress.update(task, description="[green]Training completed successfully!")
                        console.print(f"[green]✓ Training {training_id} completed successfully![/green]")
                        return status
                    elif status == 'failed':
                        progress.update(task, description="[red]Training failed")
                        error_msg = status_info.get('error', 'Unknown error')
                        console.print(f"[red]✗ Training {training_id} failed: {error_msg}[/red]")
                        return status
                    elif status == 'canceled':
                        progress.update(task, description="[yellow]Training canceled")
                        console.print(f"[yellow]Training {training_id} was canceled[/yellow]")
                        return status
                    else:
                        progress.update(task, description=f"Status: {status}")
                    
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    console.print(f"\n[yellow]Monitoring stopped. Training {training_id} continues in background.[/yellow]")
                    return "monitoring_stopped"
                except Exception as e:
                    console.print(f"[red]Error monitoring training: {e}[/red]")
                    return "error"
    
    def download_result(self, training_id: str, output_dir: str = "./") -> Optional[str]:
        """Download and extract training result including captions."""
        try:
            status_info = self.get_training_status(training_id)
            
            if status_info['status'] != 'succeeded':
                console.print(f"[red]Cannot download: Training status is '{status_info['status']}'[/red]")
                return None
            
            output_url = status_info.get('output')
            if not output_url:
                console.print(f"[red]No output URL found for training {training_id}[/red]")
                return None
            
            # Get trigger word and destination for filename
            input_config = status_info.get('input', {})
            trigger_word = input_config.get('trigger_word', 'lora')
            destination = status_info.get('destination', 'model')
            model_name = destination.split('/')[-1] if '/' in destination else destination
            
            console.print(f"[yellow]Downloading training result...[/yellow]")
            
            # Download the .tar file
            import requests
            response = requests.get(output_url, stream=True)
            response.raise_for_status()
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create captions subdirectory
            captions_path = output_path / "captions"
            captions_path.mkdir(exist_ok=True)
            
            # Save and extract the tar file
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract the tar file
                with tarfile.open(tmp_file_path, 'r') as tar:
                    tar.extractall(path=output_path)
                
                # Find and rename the .safetensors file
                safetensors_files = list(output_path.glob('*.safetensors'))
                model_file_path = None
                
                if safetensors_files:
                    original_file = safetensors_files[0]
                    new_filename = f"{model_name}_{trigger_word}.safetensors"
                    new_filepath = output_path / new_filename
                    
                    # Rename if different
                    if original_file.name != new_filename:
                        original_file.rename(new_filepath)
                    
                    model_file_path = str(new_filepath)
                    console.print(f"[green]✓ Model file: {new_filepath}[/green]")
                else:
                    console.print(f"[red]No .safetensors file found in downloaded archive[/red]")
                    return None
                
                # Find and organize caption files
                caption_files = list(output_path.glob('*.txt'))
                if caption_files:
                    caption_count = 0
                    for caption_file in caption_files:
                        # Move caption files to captions subdirectory
                        new_caption_path = captions_path / caption_file.name
                        caption_file.rename(new_caption_path)
                        caption_count += 1
                    
                    console.print(f"[green]✓ Caption files: {caption_count} files moved to captions/[/green]")
                else:
                    console.print(f"[yellow]No caption files (.txt) found in training output[/yellow]")
                
                # Clean up any remaining extracted files (keep only the model in root)
                for item in output_path.iterdir():
                    if item.is_file() and item.suffix not in ['.safetensors']:
                        if item.name not in ['.env', 'config.yaml']:  # Preserve important files
                            try:
                                item.unlink()
                            except:
                                pass  # Ignore cleanup errors
                
                return model_file_path
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            console.print(f"[red]✗ Failed to download result: {e}[/red]")
            raise
    
    def display_training_status(self, training_id: str):
        """Display detailed training status in a formatted table."""
        try:
            status_info = self.get_training_status(training_id)
            
            # Create status table
            table = Table(title=f"Training Status: {training_id}")
            table.add_column("Property", style="bold blue")
            table.add_column("Value", style="white")
            
            # Add rows
            table.add_row("Status", self._format_status(status_info['status']))
            table.add_row("Destination", status_info.get('destination', 'N/A'))
            table.add_row("Created", str(status_info.get('created_at', 'N/A')))
            
            if status_info.get('started_at'):
                table.add_row("Started", str(status_info['started_at']))
            
            if status_info.get('completed_at'):
                table.add_row("Completed", str(status_info['completed_at']))
            
            if status_info.get('error'):
                table.add_row("Error", f"[red]{status_info['error']}[/red]")
            
            # Add input parameters
            input_config = status_info.get('input', {})
            if input_config:
                table.add_row("Trigger Word", input_config.get('trigger_word', 'N/A'))
                table.add_row("Steps", str(input_config.get('steps', 'N/A')))
                table.add_row("Batch Size", str(input_config.get('batch_size', 'N/A')))
                table.add_row("Optimizer", input_config.get('optimizer', 'N/A'))
                table.add_row("Learning Rate", str(input_config.get('learning_rate', 'N/A')))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]✗ Failed to display training status: {e}[/red]")
            raise
    
    def display_training_list(self, trainings: List[Dict[str, Any]]):
        """Display a list of trainings in a formatted table."""
        if not trainings:
            console.print("[yellow]No training jobs found[/yellow]")
            return
        
        table = Table(title="Recent Training Jobs")
        table.add_column("ID", style="bold")
        table.add_column("Status", style="white")
        table.add_column("Destination", style="blue")
        table.add_column("Created", style="dim")
        
        for training in trainings:
            table.add_row(
                training['id'][:12] + "...",  # Truncate ID for display
                self._format_status(training['status']),
                training.get('destination', 'N/A'),
                str(training.get('created_at', 'N/A'))[:19] if training.get('created_at') else 'N/A'
            )
        
        console.print(table)
    
    def _format_status(self, status: str) -> str:
        """Format status with appropriate colors."""
        status_colors = {
            'starting': '[yellow]starting[/yellow]',
            'processing': '[blue]processing[/blue]',
            'succeeded': '[green]succeeded[/green]',
            'failed': '[red]failed[/red]',
            'canceled': '[dim]canceled[/dim]'
        }
        return status_colors.get(status, status)
