"""
Logging utilities for the gap filling pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


from rich.logging import RichHandler
from rich.console import Console

def setup_logger(
    name: str = "gap_filling",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with Rich terminal support and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # 1. Console handler (RICH)
    if console:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            omit_repeated_times=False
        )
        console_handler.setLevel(level)
        # Formatter is not strictly needed for RichHandler as it handles formatting itself,
        # but we can set a minimal one if desired.
        logger.addHandler(console_handler)
    
    # 2. File handler (Standard)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevenir que los logs se propaguen al logger raíz (evita duplicados si el raíz tiene handlers)
    logger.propagate = False
    
    return logger


def get_experiment_logger(experiment_name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Get a logger for a specific experiment with timestamped log file.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    log_dir : str
        Directory for log files
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=f"gap_filling.{experiment_name}",
        log_file=log_file,
        level=logging.INFO,
        console=True
    )
