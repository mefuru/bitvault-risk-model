"""
Logging configuration for BitVault Risk Model.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import get_project_root


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file
        log_to_console: Whether to write logs to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("bitvault")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_dir = get_project_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"risk_model_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "bitvault") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (will be prefixed with 'bitvault.')
        
    Returns:
        Logger instance
    """
    if name == "bitvault":
        return logging.getLogger("bitvault")
    return logging.getLogger(f"bitvault.{name}")


# Initialize default logger on import
_default_logger = setup_logging(log_to_file=True, log_to_console=False)
