"""
Utility functions for SD3.5L LoRA training tool.
Common helpers and decorators.
"""

import functools
import sys
from typing import Callable, Any
from rich.console import Console

console = Console()


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle and display errors gracefully."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation canceled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]✗ Unexpected error: {e}[/red]")
            console.print("[dim]Use --help for usage information[/dim]")
            sys.exit(1)
    
    return wrapper


def validate_training_id(training_id: str) -> None:
    """Validate that training ID is properly formatted."""
    if not training_id:
        raise ValueError("Training ID cannot be empty")
    
    if len(training_id) < 8:
        raise ValueError("Training ID appears to be too short")
    
    # Basic format validation - Replicate IDs are typically alphanumeric
    if not training_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Training ID contains invalid characters")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string with ellipsis if longer than max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_user_confirmation(message: str, default: bool = False) -> bool:
    """Get user confirmation with y/n prompt."""
    from rich.prompt import Confirm
    return Confirm.ask(message, default=default)


def display_success(message: str) -> None:
    """Display success message with consistent formatting."""
    console.print(f"[green]✓ {message}[/green]")


def display_error(message: str) -> None:
    """Display error message with consistent formatting."""
    console.print(f"[red]✗ {message}[/red]")


def display_warning(message: str) -> None:
    """Display warning message with consistent formatting."""
    console.print(f"[yellow]⚠ {message}[/yellow]")


def display_info(message: str) -> None:
    """Display info message with consistent formatting."""
    console.print(f"[blue]ℹ {message}[/blue]")
