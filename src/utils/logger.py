"""Logging configuration for the application."""

import sys
from pathlib import Path

from loguru import logger

from src.config import settings

# Remove default handler
logger.remove()

# Console handler with rich formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)

# File handler for all logs
log_path = Path("logs")
log_path.mkdir(exist_ok=True)

logger.add(
    log_path / "app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# Error-specific log file
logger.add(
    log_path / "errors_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="90 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)
