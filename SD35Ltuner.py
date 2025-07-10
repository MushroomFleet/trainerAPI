#!/usr/bin/env python3
"""
SD3.5L LoRA Training CLI Tool
Main entry point for the sd35l-trainer command-line interface.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sd35l_trainer.cli import main

if __name__ == '__main__':
    main()
