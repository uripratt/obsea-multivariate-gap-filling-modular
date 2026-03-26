"""
Logging utilities for the gap filling pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "gap_filling",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file. If None, only console logging is used.
    level : int
        Logging level
    console : bool
        Whether to log to console
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
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
